"""Small M5-shaped inputs; tests do not require the Walmart download."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def m5_data(tmp_path: Path) -> Path:
    directory = tmp_path / "data"
    directory.mkdir()
    days = 168
    weekday = np.arange(days) % 7
    sales = np.stack(
        [
            10 + weekday,
            4 + weekday,
            1 + weekday,
            np.where(np.arange(days) % 3 == 0, 20, 0),
        ]
    )
    rows = pd.DataFrame(sales, columns=[f"d_{i + 1}" for i in range(days)])
    rows.insert(0, "state_id", ["CA", "TX", "WI", "CA"])
    rows.insert(0, "item_id", [f"ITEM_{i}" for i in range(1, 5)])
    rows.insert(0, "store_id", ["CA_1", "TX_1", "WI_1", "CA_1"])
    rows.insert(0, "dept_id", ["FOODS_1"] * 4)
    rows.insert(0, "cat_id", ["FOODS"] * 4)
    rows.insert(
        0,
        "id",
        [
            "ITEM_1_CA_1_evaluation",
            "ITEM_2_TX_1_evaluation",
            "ITEM_3_WI_1_evaluation",
            "ITEM_4_CA_1_evaluation",
        ],
    )
    rows.to_csv(directory / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(
        {
            "d": [f"d_{i + 1}" for i in range(days)],
            "date": pd.date_range("2015-01-01", periods=days).strftime("%Y-%m-%d"),
            "wm_yr_wk": 1 + np.arange(days) // 7,
            "event_name_1": ["Holiday" if i % 30 == 0 else None for i in range(days)],
            "event_name_2": [None] * days,
            "snap_CA": (weekday < 2).astype(int),
            "snap_TX": (weekday > 4).astype(int),
            "snap_WI": np.zeros(days, dtype=int),
        }
    ).to_csv(directory / "calendar.csv", index=False)
    pd.DataFrame(
        [
            {"store_id": store, "item_id": f"ITEM_{i}", "wm_yr_wk": week, "sell_price": i + 1.0}
            for i, store in enumerate(["CA_1", "TX_1", "WI_1", "CA_1"], start=1)
            for week in range(1, 25)
        ]
    ).to_csv(directory / "sell_prices.csv", index=False)
    return directory
