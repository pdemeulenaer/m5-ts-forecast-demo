"""Compare Conv1D, XGBoost, and the previous-week baseline in sales units."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from m5_forecast.adjustment import adjust_totals
from m5_forecast.data import HORIZON, PreparedData, SalesWindows
from m5_forecast.model import DemandCNN


@torch.inference_mode()
def predict(
    model: DemandCNN,
    data: PreparedData,
    origins: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Predict [series, origin, horizon], restoring the training-fitted unit scale."""
    model.eval()
    loader = DataLoader(SalesWindows(data, origins), batch_size=batch_size)
    batches = []
    for history, calendar, series, _ in loader:
        output = model(history.to(device), calendar.to(device), series.to(device))
        batches.append(output.cpu().numpy())
    normalized = np.concatenate(batches).reshape(len(data.series_ids), len(origins), HORIZON)
    forecast = normalized * data.scales[:, None, None]
    return adjust_totals(forecast, data, origins, model.reconcile_alpha)


def seasonal_baseline(sales: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Repeat the seven observations immediately before each forecast date."""
    return np.stack([sales[:, origin - HORIZON : origin] for origin in origins], axis=1)


def backtest(
    model: DemandCNN,
    data: PreparedData,
    origins: np.ndarray,
    device: torch.device,
    xgboost_forecasts: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Report unit MAE over every series, forecast origin, and horizon day."""
    forecasts = {
        "cnn": predict(model, data, origins, device),
        "baseline": seasonal_baseline(data.sales, origins),
    }
    if xgboost_forecasts is not None:
        forecasts["xgboost"] = xgboost_forecasts
    actual = np.stack([data.sales[:, origin : origin + HORIZON] for origin in origins], axis=1)
    for name, values in forecasts.items():
        if values.shape != actual.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name} forecast shape or values.")
    errors = {name: np.abs(values - actual) for name, values in forecasts.items()}
    by_date = pd.DataFrame(
        {
            "forecast_date": data.dates[origins],
            **{f"{name}_mae": values.mean(axis=(0, 2)) for name, values in errors.items()},
        }
    )
    records = []
    for i, series_id in enumerate(data.series_ids):
        for j, origin in enumerate(origins):
            for h in range(HORIZON):
                records.append(
                    {
                        "series_id": series_id,
                        "forecast_date": data.dates[origin],
                        "date": data.dates[origin + h],
                        "horizon_day": h + 1,
                        "actual": float(actual[i, j, h]),
                        **{name: float(values[i, j, h]) for name, values in forecasts.items()},
                    }
                )
    summary = {
        **{f"{name}_mae": float(values.mean()) for name, values in errors.items()},
        "num_series": len(data.series_ids),
        "num_forecast_dates": len(origins),
        "num_predictions": int(actual.size),
    }
    return pd.DataFrame(records), by_date, summary
