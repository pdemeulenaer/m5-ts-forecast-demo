"""Use observed weekly totals to reduce shared bias while retaining model product shares."""

import numpy as np

from m5_forecast.data import HORIZON, PreparedData


def adjust_totals(
    forecast: np.ndarray, data: PreparedData, origins: np.ndarray, alpha: float
) -> np.ndarray:
    """Move each forecast day's total toward the previous week's observed total.

    Alpha is selected on validation, then fixed. No future actuals are used.
    With alpha=1 totals match the baseline, while each model retains its item shares.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("Total adjustment alpha must be between zero and one.")
    if alpha == 0:
        return forecast
    baseline = np.stack([data.sales[:, o - HORIZON : o] for o in origins], axis=1)
    total = forecast.sum(axis=0, keepdims=True)
    desired = (1 - alpha) * total + alpha * baseline.sum(axis=0, keepdims=True)
    ratio = np.divide(desired, total, out=np.ones_like(desired), where=total > 0)
    return np.where(total > 0, forecast * ratio, alpha * baseline)
