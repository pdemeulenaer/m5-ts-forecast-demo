"""Run a small, fixed set of improvements; select on validation WRMSSE, then test once."""

import argparse
import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

from m5_forecast.adjustment import adjust_totals
from m5_forecast.artifacts import load_artifacts, save_artifacts
from m5_forecast.data import HORIZON, SalesWindows
from m5_forecast.evaluate import backtest, predict, seasonal_baseline
from m5_forecast.metrics import WRMSSEScorer, load_revenue, refresh_metrics
from m5_forecast.model import DemandCNN
from m5_forecast.xgboost_model import predict_xgboost, train_xgboost


def train_residual_cnn(
    data, train_origins, train_scorer, val_origins, val_scorer, epochs=20, threads=4, seed=42
):
    """Train a seasonal residual CNN on complete series batches and hierarchical error."""
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    model = DemandCNN(len(data.series_ids), seasonal_residual=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    samples = list(SalesWindows(data, train_origins))
    inputs = [torch.stack(values) for values in zip(*samples)]
    count, dates = len(data.series_ids), len(train_origins)
    inputs = [values.reshape(count, dates, *values.shape[1:]) for values in inputs]
    aggregation = torch.tensor(train_scorer.aggregation, dtype=torch.float32)
    actual = torch.tensor(train_scorer.actual, dtype=torch.float32)
    scales = torch.tensor(train_scorer.scales, dtype=torch.float32)
    weights = torch.tensor(train_scorer.weights, dtype=torch.float32)
    unit_scales = torch.from_numpy(data.scales)[:, None, None]
    best_score, best_weights, best_epoch = float("inf"), None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in torch.randperm(dates).split(4):
            past, calendar, series, _ = [values[:, batch] for values in inputs]
            output = model(past.flatten(0, 1), calendar.flatten(0, 1), series.flatten())
            raw = output.reshape(count, len(batch), HORIZON) * unit_scales
            aggregate = (aggregation @ raw.flatten(1)).reshape(-1, len(batch), HORIZON)
            mse = (aggregate - actual[:, batch]).square().mean(dim=2)
            # Every batch contains all selected series at the same origins, so the
            # loss includes real store/category/total errors, not isolated item errors.
            loss = (torch.sqrt(mse / scales[:, batch] + 1e-8) * weights[:, batch]).sum(0).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        score = val_scorer.score(predict(model, data, val_origins, torch.device("cpu")))
        history.append({"epoch": epoch, "validation_wrmsse": score})
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_weights = copy.deepcopy(model.state_dict())
        print(f"Residual CNN epoch {epoch:02d}: validation WRMSSE={score:.4f}", flush=True)
    model.load_state_dict(best_weights)
    return model, {"best_epoch": best_epoch, "best_validation_wrmsse": best_score}, history


def select_adjustment(forecast, data, origins, scorer):
    """Choose from five predeclared total adjustments using validation only."""
    candidates = [
        (scorer.score(adjust_totals(forecast, data, origins, alpha)), alpha)
        for alpha in (0, 0.25, 0.5, 0.75, 1)
    ]
    return min(candidates)


def run_improvements(args):
    """Compare old/new models on validation, freeze choices, and save the final backtest."""
    if min(args.epochs, args.xgb_rounds, args.threads) < 1:
        raise ValueError("Epochs, boosting rounds, and threads must be positive.")
    torch.set_num_threads(args.threads)
    old_cnn, data, metadata = load_artifacts(args.artifacts_dir)
    old_trees = xgb.Booster(params={"nthread": args.threads})
    old_trees.load_model(args.artifacts_dir / "xgboost.json")
    old_cnn.reconcile_alpha = 0
    old_trees.set_attr(reconcile_alpha="0")
    config = metadata["run_config"].copy()
    train_origins = data.split.origins("train", config["train_stride"])
    val_origins = data.split.origins("validation")
    context = load_revenue(data, args.data_dir, np.r_[train_origins, val_origins])
    if context is None:
        raise ValueError(
            "Improvement selection requires the three local M5 CSVs, including prices."
        )
    train_scorer = WRMSSEScorer(data, train_origins, *context)
    val_scorer = WRMSSEScorer(data, val_origins, *context)
    records = [
        {
            "candidate": "seasonal baseline",
            "validation_wrmsse": val_scorer.score(seasonal_baseline(data.sales, val_origins)),
            "alpha": 0,
        }
    ]
    cnn_candidates, tree_candidates = [], []

    def consider(name, model, is_cnn, details):
        forecast = (
            predict(model, data, val_origins, torch.device("cpu"))
            if is_cnn
            else predict_xgboost(model, data, val_origins)
        )
        score, alpha = select_adjustment(forecast, data, val_origins, val_scorer)
        records.append(
            {
                "candidate": name,
                "raw_validation_wrmsse": val_scorer.score(forecast),
                "validation_wrmsse": score,
                "alpha": alpha,
            }
        )
        (cnn_candidates if is_cnn else tree_candidates).append((score, name, model, alpha, details))
        print(f"{name}: validation WRMSSE={score:.4f}, total adjustment={alpha}", flush=True)

    consider("original CNN", old_cnn, True, {})
    consider(
        "original XGBoost",
        old_trees,
        False,
        json.loads((args.artifacts_dir / "xgboost_config.json").read_text()),
    )
    cnn, cnn_details, history = train_residual_cnn(
        data,
        train_origins,
        train_scorer,
        val_origins,
        val_scorer,
        args.epochs,
        args.threads,
        config["seed"],
    )
    consider("seasonal residual CNN", cnn, True, cnn_details)
    for objective in ("reg:squarederror", "reg:tweedie"):
        print(f"Training enhanced XGBoost ({objective})...", flush=True)
        trees, details = train_xgboost(
            data,
            stride=config["train_stride"],
            rounds=args.xgb_rounds,
            threads=args.threads,
            seed=config["seed"],
            enhanced=True,
            objective=objective,
            scorer=val_scorer,
        )
        consider(f"enhanced XGBoost {objective}", trees, False, details)

    cnn_score, cnn_name, cnn, cnn_alpha, cnn_details = min(cnn_candidates, key=lambda row: row[0])
    tree_score, tree_name, trees, tree_alpha, tree_details = min(
        tree_candidates, key=lambda row: row[0]
    )
    cnn.reconcile_alpha = cnn_alpha
    trees.set_attr(reconcile_alpha=str(tree_alpha))
    output = args.output_dir or args.artifacts_dir
    if output.resolve() == args.artifacts_dir.resolve():
        backup = output / (
            "before-improvements-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        backup.mkdir()
        for path in output.iterdir():
            if path.is_file():
                shutil.copy2(path, backup / path.name)
        print(f"Original artifacts preserved in {backup}", flush=True)
    config.update(
        selection_metric="validation_wrmsse",
        selected_cnn=cnn_name,
        selected_xgboost=tree_name,
        data_dir=str(args.data_dir),
        **cnn_details,
    )
    config["adjusted_validation_wrmsse"] = cnn_score
    if cnn_name == "seasonal residual CNN":
        config.update(epochs=args.epochs, lr=0.0005, training_loss="hierarchical_wrmsse")
        config.pop("best_validation_mae", None)
    # Test forecasts are produced only after both model choices and adjustments are fixed.
    origins = data.split.origins("test")
    forecasts, _, metrics = backtest(
        cnn, data, origins, torch.device("cpu"), predict_xgboost(trees, data, origins)
    )
    save_artifacts(output, cnn, data, config, metrics)
    trees.save_model(output / "xgboost.json")
    tree_details.update(reconcile_alpha=tree_alpha, adjusted_validation_wrmsse=tree_score)
    (output / "xgboost_config.json").write_text(json.dumps(tree_details, indent=2) + "\n")
    forecasts.to_csv(output / "test_forecasts.csv", index=False)
    metrics = refresh_metrics(output, data, args.data_dir)
    report = {
        "selection_metric": "validation_wrmsse",
        "validation_candidates": records,
        "selected_cnn": cnn_name,
        "selected_xgboost": tree_name,
        "cnn_validation_wrmsse": cnn_score,
        "xgboost_validation_wrmsse": tree_score,
        "before_test_metrics": metadata["test_metrics"],
        "after_test_metrics": metrics,
        "settings": {
            "epochs": args.epochs,
            "xgb_rounds": args.xgb_rounds,
            "threads": args.threads,
            "seed": config["seed"],
        },
    }
    (output / "improvement_results.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "residual_training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    if cnn_name == "seasonal residual CNN":
        (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    """Improve an existing run; optionally write a separate output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--xgb-rounds", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        run_improvements(args)
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
