"""M5 RMSSE and twelve-level WRMSSE, evaluated on the demo's selected series."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from m5_forecast.artifacts import load_artifacts
from m5_forecast.data import HORIZON, PreparedData

LEVELS = [
    ("Total", []),
    ("State", ["state_id"]),
    ("Store", ["store_id"]),
    ("Category", ["cat_id"]),
    ("Department", ["dept_id"]),
    ("State-category", ["state_id", "cat_id"]),
    ("State-department", ["state_id", "dept_id"]),
    ("Store-category", ["store_id", "cat_id"]),
    ("Store-department", ["store_id", "dept_id"]),
    ("Item", ["item_id"]),
    ("Item-state", ["item_id", "state_id"]),
    ("Item-store", ["id"]),
]


def rmsse_scale(history: np.ndarray) -> float:
    """Mean squared daily change after the first nonzero sale; NaN if undefined."""
    history = np.asarray(history, dtype=np.float64)
    active = np.flatnonzero(history > 0)
    if len(active) == 0 or len(history) - active[0] < 2:
        return float("nan")
    scale = float(np.square(np.diff(history[active[0] :])).mean())
    return scale if scale > 0 else float("nan")


def rmsse(actual: np.ndarray, forecast: np.ndarray, history: np.ndarray) -> float:
    """Scale forecast RMSE by the historical one-day naive RMSE."""
    error = np.asarray(forecast, dtype=np.float64) - actual
    return float(np.sqrt(np.square(error).mean() / rmsse_scale(history)))


def load_revenue(
    data: PreparedData, data_dir: Path, origins: np.ndarray | None = None
) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Align hierarchy and observed dollar sales; never infer missing selling prices."""
    names = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]
    if not all((data_dir / name).is_file() for name in names):
        return None
    columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    hierarchy = pd.read_csv(data_dir / names[0], usecols=columns).set_index("id")
    hierarchy = hierarchy.loc[data.series_ids].reset_index()
    if hierarchy.isna().any().any():
        raise ValueError("Hierarchy identifiers must not be missing.")
    days = [f"d_{i + 1}" for i in range(data.split.total_days)]
    weeks = pd.read_csv(data_dir / names[1]).set_index("d").loc[days, "wm_yr_wk"].to_numpy()
    # Only observed prices before the latest supplied origin are needed for weights.
    origins = data.split.origins("test") if origins is None else origins
    cutoff = int(origins.max())
    pairs = pd.MultiIndex.from_frame(hierarchy[["store_id", "item_id"]])
    selected = []
    for chunk in pd.read_csv(data_dir / names[2], chunksize=100_000):
        keys = pd.MultiIndex.from_frame(chunk[["store_id", "item_id"]])
        selected.append(chunk[keys.isin(pairs) & chunk.wm_yr_wk.isin(weeks[:cutoff])])
    prices = pd.concat(selected).set_index(["store_id", "item_id", "wm_yr_wk"])["sell_price"]
    if prices.index.duplicated().any():
        raise ValueError("Selling prices must be unique per store, item, and week.")
    revenue = np.zeros_like(data.sales, dtype=np.float64)
    # Only the past 28-day weighting windows need complete prices.
    needed = np.zeros(data.split.total_days, dtype=bool)
    for origin in origins:
        needed[origin - 28 : origin] = True
    for series, row in hierarchy.iterrows():
        keys = pd.MultiIndex.from_arrays(
            [
                np.repeat(row.store_id, needed.sum()),
                np.repeat(row.item_id, needed.sum()),
                weeks[needed],
            ]
        )
        values = prices.reindex(keys).to_numpy(dtype=np.float64)
        sold = data.sales[series, needed] > 0
        if (~np.isfinite(values[sold])).any() or (values[sold] <= 0).any():
            raise ValueError(f"Missing or invalid historical price for selling series {row.id}.")
        revenue[series, needed] = data.sales[series, needed] * np.where(sold, values, 0)
    return hierarchy, revenue


class WRMSSEScorer:
    """Cache hierarchy sums, past-only scales, and revenue weights for repeated validation."""

    def __init__(
        self, data: PreparedData, origins: np.ndarray, hierarchy: pd.DataFrame, revenue: np.ndarray
    ):
        rows, level_sizes = [], []
        for _, columns in LEVELS:
            groups = (
                list(hierarchy.groupby(columns, sort=False).indices.values())
                if columns
                else [np.arange(len(hierarchy))]
            )
            level_sizes.append(len(groups))
            for indices in groups:
                row = np.zeros(len(data.series_ids))
                row[indices] = 1
                rows.append(row)
        self.aggregation = np.asarray(rows)
        sales = self.aggregation @ data.sales.astype(np.float64)
        dollars = self.aggregation @ revenue
        self.scales = np.array([[rmsse_scale(series[:o]) for o in origins] for series in sales])
        self.weights = np.stack([dollars[:, o - 28 : o].sum(axis=1) for o in origins], axis=1)
        offset = 0
        for size in level_sizes:
            weights = self.weights[offset : offset + size]
            totals = weights.sum(axis=0)
            if (totals <= 0).any():
                raise ValueError(
                    "WRMSSE selection requires positive recent revenue at every origin."
                )
            weights /= totals * len(LEVELS)
            offset += size
        invalid = ~np.isfinite(self.scales)
        if (invalid & (self.weights > 0)).any():
            raise ValueError("WRMSSE selection is undefined for constant positive-weight series.")
        self.scales[invalid] = 1  # zero-weight groups contribute zero
        self.actual = np.stack([sales[:, o : o + HORIZON] for o in origins], axis=1)

    def score(self, forecast: np.ndarray) -> float:
        """Mean WRMSSE across the cached forecast dates."""
        aggregate = (self.aggregation @ forecast.reshape(forecast.shape[0], -1)).reshape(
            self.actual.shape
        )
        mse = np.square(aggregate - self.actual).mean(axis=2)
        return float((np.sqrt(mse / self.scales) * self.weights).sum(axis=0).mean())


def hierarchical_wrmsse(
    data: PreparedData,
    hierarchy: pd.DataFrame,
    revenue: np.ndarray,
    forecast: np.ndarray,
    origin: int,
) -> tuple[float, list[dict]]:
    """Sum bottom forecasts into all levels, normalize revenue within each, then average.

    Historical scales and the last 28 days' dollar weights use observations strictly
    before this origin. Positively weighted constant series make a score undefined.
    """
    rows = []
    actual = data.sales[:, origin : origin + forecast.shape[1]]
    dollars = revenue[:, origin - 28 : origin].sum(axis=1)
    for name, columns in LEVELS:
        groups = (
            hierarchy.groupby(columns, sort=False).indices.values()
            if columns
            else [np.arange(len(hierarchy))]
        )
        weights, scores = [], []
        for indices in groups:
            weights.append(dollars[indices].sum())
            scores.append(
                rmsse(
                    actual[indices].sum(axis=0),
                    forecast[indices].sum(axis=0),
                    data.sales[indices, :origin].sum(axis=0),
                )
            )
        weights, scores = np.asarray(weights), np.asarray(scores)
        positive = weights > 0
        score = (
            float(np.sum(weights[positive] * scores[positive]) / weights.sum())
            if positive.any()
            else float("nan")
        )
        rows.append({"level": name, "num_series": len(scores), "wrmsse": score})
    return float(np.mean([row["wrmsse"] for row in rows])), rows


def finite_or_none(value: float) -> float | None:
    """Keep saved JSON valid when a constant series makes scaled error undefined."""
    return float(value) if np.isfinite(value) else None


def refresh_metrics(directory: Path, data: PreparedData, data_dir: Path) -> dict:
    """Score saved forecasts without retraining; write overall, date, and series reports."""
    origins = data.split.origins("test")
    index = pd.MultiIndex.from_product(
        [data.series_ids, data.dates[origins], range(1, HORIZON + 1)],
        names=["series_id", "forecast_date", "horizon_day"],
    )
    saved = pd.read_csv(directory / "test_forecasts.csv").set_index(index.names).reindex(index)
    actual = np.stack([data.sales[:, o : o + HORIZON] for o in origins], axis=1).astype(np.float64)
    if not np.allclose(saved.actual.to_numpy().reshape(actual.shape), actual):
        raise ValueError("Saved forecast actuals do not match the selected data.")
    scales = np.array([[rmsse_scale(sales[:o]) for o in origins] for sales in data.sales])
    context = load_revenue(data, data_dir)
    summary = {
        "num_series": len(data.series_ids),
        "num_forecast_dates": len(origins),
        "num_predictions": int(actual.size),
        "horizon": HORIZON,
        "wrmsse_scope": "12 aggregation levels of selected demo series; 7-day forecasts",
        "wrmsse_status": "available" if context is not None else "sell_prices.csv required",
    }
    by_date = pd.DataFrame({"forecast_date": data.dates[origins]})
    by_series = pd.DataFrame({"series_id": data.series_ids})
    level_rows = []
    for method in ("cnn", "xgboost", "baseline"):
        forecast = saved[method].to_numpy().reshape(actual.shape)
        if not np.isfinite(forecast).all():
            raise ValueError(f"Missing or invalid {method} forecasts.")
        errors = forecast - actual
        scaled_errors = np.sqrt(np.square(errors).mean(axis=2) / scales)
        summary[f"{method}_mae"] = float(np.abs(errors).mean())
        summary[f"{method}_rmsse"] = finite_or_none(scaled_errors.mean())
        by_date[f"{method}_mae"] = np.abs(errors).mean(axis=(0, 2))
        by_date[f"{method}_rmsse"] = scaled_errors.mean(axis=0)
        by_series[f"{method}_mae"] = np.abs(errors).mean(axis=(1, 2))
        by_series[f"{method}_rmsse"] = scaled_errors.mean(axis=1)
        weighted = []
        for j, origin in enumerate(origins):
            if context is not None:
                score, levels = hierarchical_wrmsse(data, *context, forecast[:, j], int(origin))
                weighted.append(score)
                level_rows.extend(
                    {"method": method, "forecast_date": data.dates[origin], **row} for row in levels
                )
            else:
                weighted.append(float("nan"))
        by_date[f"{method}_wrmsse"] = weighted
        summary[f"{method}_wrmsse"] = finite_or_none(np.mean(weighted))
    by_date.to_csv(directory / "mae_by_date.csv", index=False)
    if context is not None and any(
        summary[f"{name}_wrmsse"] is None for name in ("cnn", "xgboost", "baseline")
    ):
        summary["wrmsse_status"] = "undefined: constant history or zero revenue"
    by_series.to_csv(directory / "metrics_by_series.csv", index=False)
    pd.DataFrame(
        level_rows, columns=["method", "forecast_date", "level", "num_series", "wrmsse"]
    ).to_csv(directory / "wrmsse_by_level.csv", index=False)
    (directory / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    metadata_path = directory / "preprocessing.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["test_metrics"] = summary
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return summary


def main() -> None:
    """Refresh M5 metrics for an existing saved run; weights remain unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    try:
        _, data, _ = load_artifacts(args.artifacts_dir)
        print(json.dumps(refresh_metrics(args.artifacts_dir, data, args.data_dir), indent=2))
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
