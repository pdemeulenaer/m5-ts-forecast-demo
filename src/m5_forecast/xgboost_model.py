"""One shared tree model, with forecast horizon included as a feature."""

import numpy as np
import xgboost as xgb

from m5_forecast.adjustment import adjust_totals
from m5_forecast.data import HISTORY, HORIZON, PreparedData


def make_features(
    data: PreparedData,
    origins: np.ndarray,
    series_indices: list[int] | None = None,
    enhanced: bool = False,
) -> np.ndarray:
    """Build rows in series/origin/horizon order, using no future sales.

    Each row contains 56 normalized sales lags, a one-hot series ID, horizon day,
    and the target date's known calendar features. Identity uses the saved ID order.
    """
    indices = range(len(data.series_ids)) if series_indices is None else series_indices
    features = []
    for series in indices:
        normalized = data.sales[series] / data.scales[series]
        past = np.stack([normalized[o - HISTORY : o] for o in origins])
        identity = np.zeros((len(origins) * HORIZON, len(data.series_ids)), dtype=np.float32)
        identity[:, series] = 1
        calendar = np.stack([data.calendar[series, o : o + HORIZON] for o in origins])
        summaries = []
        if enhanced:
            for length in (7, 14, 28, 56):
                recent = past[:, -length:]
                summaries.extend(
                    [recent.mean(axis=1), recent.std(axis=1), (recent == 0).mean(axis=1)]
                )
            summaries.append(np.full(len(origins), data.scales[series]))
            # Matching weekday observations for each horizon, all before the origin.
            weekday_lags = np.stack(
                [past[:, -7:], past[:, -14:-7], past[:, -28:].reshape(-1, 4, 7).mean(axis=1)],
                axis=2,
            )
        features.append(
            np.column_stack(
                [
                    np.repeat(past, HORIZON, axis=0),
                    identity,
                    np.tile(np.arange(1, HORIZON + 1, dtype=np.float32), len(origins)),
                    calendar.reshape(-1, calendar.shape[-1]),
                    *(
                        [
                            np.repeat(np.column_stack(summaries), HORIZON, axis=0),
                            weekday_lags.reshape(-1, 3),
                        ]
                        if enhanced
                        else []
                    ),
                ]
            )
        )
    return np.concatenate(features).astype(np.float32)


def predict_xgboost(
    model: xgb.Booster,
    data: PreparedData,
    origins: np.ndarray,
    series_indices: list[int] | None = None,
) -> np.ndarray:
    """Predict [series, origin, horizon] in units, clipping negative predictions."""
    indices = list(range(len(data.series_ids))) if series_indices is None else series_indices
    alpha = float(model.attr("reconcile_alpha") or 0)
    if alpha:
        indices = list(range(len(data.series_ids)))
    features = make_features(data, origins, indices, enhanced=model.attr("enhanced") == "true")
    normalized = model.inplace_predict(features).reshape(len(indices), len(origins), HORIZON)
    forecast = np.maximum(normalized, 0) * data.scales[indices, None, None]
    if alpha:
        forecast = adjust_totals(forecast, data, origins, alpha)
        if series_indices is not None:
            forecast = forecast[series_indices]
    return forecast


def train_xgboost(
    data: PreparedData,
    stride: int = 7,
    rounds: int = 300,
    threads: int = 4,
    seed: int = 42,
    enhanced: bool = False,
    objective: str = "reg:squarederror",
    scorer=None,
) -> tuple[xgb.Booster, dict]:
    """Train on training windows; use validation unit MAE to stop and retain trees."""
    if min(stride, rounds, threads) < 1:
        raise ValueError("XGBoost stride, rounds, and threads must be positive.")
    matrices = {}
    for period in ("train", "validation"):
        origins = data.split.origins(period, stride if period == "train" else HORIZON)
        targets = np.stack([data.sales[:, o : o + HORIZON] for o in origins], axis=1)
        normalized = targets / data.scales[:, None, None]
        matrices[period] = xgb.DMatrix(
            make_features(data, origins, enhanced=enhanced),
            label=normalized.reshape(-1),
            nthread=threads,
        )
    val_scales = np.repeat(data.scales, len(data.split.origins("validation")) * HORIZON)

    def unit_mae(predictions, matrix):
        if scorer is not None:
            raw = (np.maximum(predictions, 0) * val_scales).reshape(
                len(data.series_ids), -1, HORIZON
            )
            return "wrmsse", scorer.score(raw)
        errors = np.abs(np.maximum(predictions, 0) - matrix.get_label()) * val_scales
        return "unit_mae", float(errors.mean())

    parameters = {
        "objective": objective,
        "tree_method": "hist",
        "device": "cpu",
        "max_depth": 5,
        "eta": 0.05,
        "max_bin": 64,
        "nthread": threads,
        "seed": seed,
        "disable_default_eval_metric": True,
    }
    if objective == "reg:tweedie":
        parameters["tweedie_variance_power"] = 1.5
    model = xgb.train(
        parameters,
        matrices["train"],
        num_boost_round=rounds,
        evals=[(matrices["validation"], "validation")],
        custom_metric=unit_mae,
        maximize=False,
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    config = {
        "rounds": rounds,
        "best_round": model.best_iteration + 1,
        "best_validation_mae": float(model.best_score),
        "train_stride": stride,
        "threads": threads,
        "seed": seed,
        "parameters": parameters,
        "enhanced": enhanced,
        "selection_metric": "wrmsse" if scorer is not None else "mae",
    }
    # Keep only the best validation prefix; saved and live predictions use identical trees.
    if scorer is not None:
        config["best_validation_wrmsse"] = config.pop("best_validation_mae")
    model = model[: config["best_round"]]
    model.set_attr(enhanced="true" if enhanced else "false")
    return model, config
