"""Load M5 data and build chronological forecasting windows."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

HISTORY = 56
HORIZON = 7
CALENDAR_FEATURES = [
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "event_1",
    "event_2",
    "snap",
]


@dataclass
class Split:
    """Exclusive boundaries for training, validation, and test days."""

    train_end: int
    val_end: int
    total_days: int

    @classmethod
    def create(cls, total_days: int, val_days: int = 56, test_days: int = 56):
        """Reserve the latest days for validation and test targets."""
        if val_days < HORIZON or test_days < 2 * HORIZON:
            raise ValueError("Use at least 7 validation days and 14 test days.")
        train_end = total_days - val_days - test_days
        if train_end < HISTORY + HORIZON:
            raise ValueError("Not enough training days for a 56 -> 7 window.")
        return cls(train_end, total_days - test_days, total_days)

    def origins(self, period: str, stride: int = HORIZON) -> np.ndarray:
        """Return first forecast-day indices; all seven targets stay in one split."""
        if stride < 1:
            raise ValueError("Stride must be positive.")
        bounds = {
            "train": (HISTORY, self.train_end),
            "validation": (self.train_end, self.val_end),
            "test": (self.val_end, self.total_days),
        }
        start, end = bounds[period]
        return np.arange(start, end - HORIZON + 1, stride)


@dataclass
class PreparedData:
    """Selected sales, known calendar features, and training-fitted scales."""

    sales: np.ndarray  # [series, day], raw units
    calendar: np.ndarray  # [series, day, feature]
    dates: np.ndarray
    series_ids: list[str]  # embedding indices follow this order
    scales: np.ndarray
    split: Split


def calendar_features(calendar: pd.DataFrame, states: list[str]) -> np.ndarray:
    """Encode dates, event presence, and state-specific SNAP without fitting."""
    dates = pd.to_datetime(calendar["date"])
    weekday = 2 * np.pi * dates.dt.dayofweek.to_numpy() / 7
    month = 2 * np.pi * (dates.dt.month.to_numpy() - 1) / 12
    common = np.column_stack(
        [
            np.sin(weekday),
            np.cos(weekday),
            np.sin(month),
            np.cos(month),
            calendar["event_name_1"].notna(),
            calendar["event_name_2"].notna(),
        ]
    )
    return np.stack(
        [np.column_stack([common, calendar[f"snap_{state}"].to_numpy()]) for state in states]
    ).astype(np.float32)


def prepare_data(
    data_dir: Path, num_series: int = 100, val_days: int = 56, test_days: int = 56
) -> PreparedData:
    """Select regular sellers using training days only, then load their full history.

    CSVs are read in chunks so the full M5 sales matrix need not live in memory.
    Selection ranks nonzero-day fraction, then mean sales, then ID for stable ties.
    Each series is divided by its training mean, floored at one unit.
    """
    if num_series < 1:
        raise ValueError("Number of series must be positive.")
    sales_path = data_dir / "sales_train_evaluation.csv"
    calendar_path = data_dir / "calendar.csv"
    for path in (sales_path, calendar_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Download the M5 CSVs; see README.md.")

    columns = pd.read_csv(sales_path, nrows=0).columns
    days = sorted((c for c in columns if c.startswith("d_")), key=lambda c: int(c[2:]))
    if days != [f"d_{i}" for i in range(1, len(days) + 1)]:
        raise ValueError("Sales columns must be consecutive, starting at d_1.")
    split = Split.create(len(days), val_days, test_days)
    train_days = days[: split.train_end]
    rankings = []
    for chunk in pd.read_csv(sales_path, usecols=["id", *train_days], chunksize=1000):
        values = chunk[train_days].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Training sales must be finite and nonnegative.")
        rankings.append(
            pd.DataFrame(
                {
                    "id": chunk["id"],
                    "regularity": (values > 0).mean(axis=1),
                    "mean": values.mean(axis=1),
                }
            )
        )
    ranking = pd.concat(rankings, ignore_index=True)
    if ranking["id"].duplicated().any():
        raise ValueError("Sales series IDs must be unique.")
    ranking = ranking[ranking["mean"] > 0].sort_values(
        ["regularity", "mean", "id"], ascending=[False, False, True]
    )
    if len(ranking) < num_series:
        raise ValueError(f"Requested {num_series} series; only {len(ranking)} sell in training.")
    series_ids = ranking.head(num_series)["id"].tolist()

    selected = []
    for chunk in pd.read_csv(sales_path, usecols=["id", "state_id", *days], chunksize=1000):
        selected.append(chunk[chunk["id"].isin(series_ids)])
    selected = pd.concat(selected).set_index("id").loc[series_ids]
    sales = selected[days].to_numpy(dtype=np.float32)
    if not np.isfinite(sales).all() or (sales < 0).any():
        raise ValueError("Selected sales must be finite and nonnegative.")
    scales = np.maximum(sales[:, : split.train_end].mean(axis=1), 1).astype(np.float32)

    calendar = pd.read_csv(calendar_path).set_index("d").loc[days]
    dates = pd.to_datetime(calendar["date"])
    day_steps = np.diff(dates.to_numpy(dtype="datetime64[ns]"))
    if dates.isna().any() or not (day_steps == np.timedelta64(1, "D")).all():
        raise ValueError("Calendar dates must be valid consecutive daily dates.")
    features = calendar_features(calendar, selected["state_id"].tolist())
    if not np.isfinite(features).all():
        raise ValueError("Calendar features must be finite.")
    return PreparedData(
        sales,
        features,
        dates.dt.strftime("%Y-%m-%d").to_numpy(dtype=str),
        series_ids,
        scales,
        split,
    )


class SalesWindows(Dataset):
    """Create windows on demand instead of copying the whole windowed dataset."""

    def __init__(self, data: PreparedData, origins: np.ndarray):
        self.sales = data.sales / data.scales[:, None]
        self.calendar = data.calendar
        self.origins = origins

    def __len__(self):
        return len(self.sales) * len(self.origins)

    def __getitem__(self, index):
        series, date_index = divmod(index, len(self.origins))
        origin = self.origins[date_index]
        history = self.sales[series, origin - HISTORY : origin, None]
        past_calendar = self.calendar[series, origin - HISTORY : origin]
        inputs = np.concatenate([history, past_calendar], axis=1).T.copy()
        future_calendar = self.calendar[series, origin : origin + HORIZON].copy()
        targets = self.sales[series, origin : origin + HORIZON].copy()
        return (
            torch.from_numpy(inputs),
            torch.from_numpy(future_calendar),
            torch.tensor(series, dtype=torch.long),
            torch.from_numpy(targets),
        )
