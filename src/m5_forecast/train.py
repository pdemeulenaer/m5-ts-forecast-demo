"""Train Conv1D and XGBoost, select on validation, and compare on held-out test days."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from m5_forecast.artifacts import load_artifacts, save_artifacts
from m5_forecast.data import HORIZON, SalesWindows, prepare_data
from m5_forecast.evaluate import backtest, predict
from m5_forecast.metrics import refresh_metrics
from m5_forecast.model import DemandCNN
from m5_forecast.xgboost_model import predict_xgboost, train_xgboost


def choose_device(requested: str) -> torch.device:
    """Use CUDA when available, or fail clearly when CUDA was explicitly requested."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install the cuda extra or use --device cpu.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def run_training(args: argparse.Namespace) -> dict:
    """Fit Conv1D, select its validation checkpoint, then fit and compare XGBoost."""
    if min(args.epochs, args.batch_size, args.train_stride, args.threads) < 1 or args.lr <= 0:
        raise ValueError("Epochs, batch size, stride, threads, and learning rate must be positive.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    data = prepare_data(args.data_dir, args.num_series, args.val_days, args.test_days)
    loader = DataLoader(
        SalesWindows(data, data.split.origins("train", args.train_stride)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_origins = data.split.origins("validation")
    val_actual = np.stack([data.sales[:, o : o + HORIZON] for o in val_origins], axis=1)
    model = DemandCNN(len(data.series_ids)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()
    best_mae, best_epoch, best_weights = float("inf"), 0, None
    history = []
    print(
        f"Device: {device}; series: {len(data.series_ids)}; training windows: {len(loader.dataset)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for past, calendar, series, target in loader:
            optimizer.zero_grad()
            output = model(past.to(device), calendar.to(device), series.to(device))
            loss = loss_fn(output, target.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(series)
        val_mae = float(np.abs(predict(model, data, val_origins, device) - val_actual).mean())
        if not np.isfinite(val_mae):
            raise ValueError(
                "Training produced non-finite validation MAE. Try a lower learning rate."
            )
        if val_mae < best_mae:
            best_mae, best_epoch = val_mae, epoch
            best_weights = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_scaled_mae": total_loss / len(loader.dataset),
                "validation_mae": val_mae,
            }
        )
        print(
            f"Epoch {epoch:02d}: scaled train MAE={history[-1]['train_scaled_mae']:.4f}; "
            f"validation MAE={val_mae:.3f} units"
        )

    model.load_state_dict(best_weights)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config.update(best_epoch=best_epoch, best_validation_mae=best_mae, resolved_device=str(device))
    metrics = save_comparison(args, model, data, config)
    (args.artifacts_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"Best Conv1D epoch: {best_epoch}")
    return metrics


def save_comparison(args, model, data, config, existing_metadata=None) -> dict:
    """Fit XGBoost on the same windows and save all three methods' test results."""
    print("Training shared XGBoost model on CPU...", flush=True)
    trees, tree_config = train_xgboost(
        data,
        stride=config["train_stride"],
        rounds=args.xgb_rounds,
        threads=args.threads,
        seed=config["seed"],
    )
    origins = data.split.origins("test")
    tree_forecasts = predict_xgboost(trees, data, origins)
    device = next(model.parameters()).device
    forecasts, by_date, metrics = backtest(model, data, origins, device, tree_forecasts)
    if existing_metadata is None:
        save_artifacts(args.artifacts_dir, model, data, config, metrics)
    else:
        existing_metadata["test_metrics"] = metrics
        (args.artifacts_dir / "preprocessing.json").write_text(
            json.dumps(existing_metadata, indent=2) + "\n"
        )
    trees.save_model(args.artifacts_dir / "xgboost.json")
    (args.artifacts_dir / "xgboost_config.json").write_text(
        json.dumps(tree_config, indent=2) + "\n"
    )
    forecasts.to_csv(args.artifacts_dir / "test_forecasts.csv", index=False)
    by_date.to_csv(args.artifacts_dir / "mae_by_date.csv", index=False)
    (args.artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    metrics = refresh_metrics(args.artifacts_dir, data, Path(config["data_dir"]))
    print(f"Best XGBoost round: {tree_config['best_round']}\n{by_date.to_string(index=False)}")
    print(
        "Test MAE: "
        + "; ".join(
            f"{name}={metrics[f'{name}_mae']:.3f}" for name in ("cnn", "xgboost", "baseline")
        )
        + " units"
    )
    print(f"Saved artifacts to {args.artifacts_dir}")
    return metrics


def main() -> None:
    """Command-line entry point; run with python -m m5_forecast.train."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--num-series", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--train-stride",
        type=int,
        default=7,
        help="Days between training forecast dates; use 1 for more windows.",
    )
    parser.add_argument("--val-days", type=int, default=56)
    parser.add_argument("--test-days", type=int, default=56)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--xgb-rounds", type=int, default=300, help="Maximum XGBoost boosting rounds."
    )
    parser.add_argument(
        "--xgboost-only",
        action="store_true",
        help="Add XGBoost to saved artifacts without retraining Conv1D.",
    )
    args = parser.parse_args()
    try:
        if args.xgboost_only:
            torch.set_num_threads(args.threads)
            model, data, metadata = load_artifacts(args.artifacts_dir)
            save_comparison(args, model, data, metadata["run_config"], metadata)
        else:
            run_training(args)
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
