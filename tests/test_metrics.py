"""Known-answer M5 metrics and checks against future weights and missing prices."""

import numpy as np
import pandas as pd
import pytest
import torch

from m5_forecast.artifacts import save_artifacts
from m5_forecast.data import PreparedData, Split, prepare_data
from m5_forecast.evaluate import backtest
from m5_forecast.metrics import (
    hierarchical_wrmsse,
    load_revenue,
    refresh_metrics,
    rmsse,
    rmsse_scale,
)
from m5_forecast.model import DemandCNN
from m5_forecast.xgboost_model import predict_xgboost, train_xgboost


def test_rmsse_trims_leading_zeros_and_scales_squared_errors():
    history = np.array([0, 0, 2, 4, 2])
    assert rmsse_scale(history) == pytest.approx(4)
    assert rmsse(np.array([2, 4]), np.array([4, 6]), history) == pytest.approx(1)
    assert np.isnan(rmsse_scale(np.array([0, 0, 0])))
    assert np.isnan(rmsse_scale(np.array([0, 3, 3, 3])))
    assert np.isnan(rmsse_scale(np.array([0, 0, 3])))


def test_wrmsse_aggregates_before_scoring_and_weights_all_12_levels():
    sales = np.stack([np.arange(50), 2 * np.arange(50)]).astype(np.float32)
    data = PreparedData(
        sales,
        np.zeros((2, 50, 7)),
        np.arange(50).astype(str),
        ["a", "b"],
        np.ones(2),
        Split(20, 32, 50),
    )
    hierarchy = pd.DataFrame(
        {
            "id": ["a", "b"],
            "item_id": ["a", "b"],
            "state_id": ["CA", "CA"],
            "store_id": ["CA_1", "CA_1"],
            "cat_id": ["FOODS", "FOODS"],
            "dept_id": ["FOODS_1", "FOODS_1"],
        }
    )
    revenue = np.stack([np.full(50, 4.0), np.ones(50)])
    forecast = sales[:, 32:39] + np.array([[1], [-1]])
    score, levels = hierarchical_wrmsse(data, hierarchy, revenue, forecast, origin=32)
    # Errors cancel at the first nine aggregate levels. Each of the last three
    # scores is 0.8 * RMSSE(a) + 0.2 * RMSSE(b) = 0.8 * 1 + 0.2 * 0.5 = 0.9.
    assert score == pytest.approx(3 * 0.9 / 12)
    assert len(levels) == 12
    assert levels[0]["wrmsse"] == pytest.approx(0)
    assert levels[-1]["wrmsse"] == pytest.approx(0.9)
    revenue[:, 32:] = 1_000_000
    assert hierarchical_wrmsse(data, hierarchy, revenue, forecast, 32)[0] == pytest.approx(score)
    assert np.isnan(hierarchical_wrmsse(data, hierarchy, np.zeros_like(revenue), forecast, 32)[0])


def test_price_loader_ignores_future_weeks_and_requires_prices_for_sales(m5_data):
    data = prepare_data(m5_data, num_series=3, val_days=28, test_days=28)
    _, before = load_revenue(data, m5_data)
    path = m5_data / "sell_prices.csv"
    prices = pd.read_csv(path)
    prices.loc[prices.wm_yr_wk == 24, "sell_price"] = 100_000
    prices.to_csv(path, index=False)
    _, after = load_revenue(data, m5_data)
    np.testing.assert_array_equal(before, after)
    prices = prices[~((prices.item_id == "ITEM_1") & (prices.wm_yr_wk == 20))]
    prices.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing or invalid historical price"):
        load_revenue(data, m5_data)


def test_metric_reports_cover_same_series_dates_and_do_not_retrain(m5_data, tmp_path):
    data = prepare_data(m5_data, num_series=3, val_days=28, test_days=28)
    model = DemandCNN(3)
    trees, _ = train_xgboost(data, rounds=2, threads=1)
    origins = data.split.origins("test")
    rows, _, original = backtest(
        model, data, origins, torch.device("cpu"), predict_xgboost(trees, data, origins)
    )
    directory = tmp_path / "artifacts"
    save_artifacts(directory, model, data, {}, original)
    rows.to_csv(directory / "test_forecasts.csv", index=False)
    checkpoint = (directory / "model.pt").read_bytes()
    metrics = refresh_metrics(directory, data, m5_data)
    assert (directory / "model.pt").read_bytes() == checkpoint
    assert metrics["num_predictions"] == original["num_predictions"]
    dates = pd.read_csv(directory / "mae_by_date.csv")
    series = pd.read_csv(directory / "metrics_by_series.csv")
    levels = pd.read_csv(directory / "wrmsse_by_level.csv")
    for method in ("cnn", "xgboost", "baseline"):
        assert metrics[f"{method}_mae"] == pytest.approx(original[f"{method}_mae"])
        assert metrics[f"{method}_rmsse"] == pytest.approx(series[f"{method}_rmsse"].mean())
        assert metrics[f"{method}_wrmsse"] == pytest.approx(dates[f"{method}_wrmsse"].mean())
    assert metrics["baseline_wrmsse"] == pytest.approx(0)
    assert len(levels) == 3 * 4 * 12
    (m5_data / "sell_prices.csv").unlink()
    unavailable = refresh_metrics(directory, data, m5_data)
    assert unavailable["cnn_wrmsse"] is None
    assert unavailable["cnn_rmsse"] == metrics["cnn_rmsse"]
    assert unavailable["wrmsse_status"] == "sell_prices.csv required"
