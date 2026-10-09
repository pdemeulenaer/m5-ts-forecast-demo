"""Check tree features, held-out isolation, persistence, and comparable unit MAE."""

import copy

import numpy as np
import pytest
import torch
import xgboost as xgb

from m5_forecast.data import HISTORY, HORIZON, prepare_data
from m5_forecast.evaluate import backtest
from m5_forecast.model import DemandCNN
from m5_forecast.xgboost_model import make_features, predict_xgboost, train_xgboost


def test_tree_features_ignore_future_sales(m5_data):
    data = prepare_data(m5_data, num_series=3, val_days=28, test_days=28)
    origin = data.split.origins("test")[0]
    before = make_features(data, np.array([origin]))
    changed = copy.deepcopy(data)
    changed.sales[:, origin:] = 50_000
    np.testing.assert_array_equal(before, make_features(changed, np.array([origin])))
    np.testing.assert_allclose(
        before[:7, :HISTORY],
        np.tile(data.sales[0, origin - HISTORY : origin] / data.scales[0], (HORIZON, 1)),
    )
    np.testing.assert_array_equal(before[:, HISTORY : HISTORY + 3], np.repeat(np.eye(3), 7, axis=0))
    np.testing.assert_array_equal(before[:7, HISTORY + 3], np.arange(1, 8))
    np.testing.assert_array_equal(before[:7, -7:], data.calendar[0, origin : origin + HORIZON])


def test_tree_training_ignores_test_sales_and_roundtrip_scores(m5_data, tmp_path):
    data = prepare_data(m5_data, num_series=3, val_days=28, test_days=28)
    trees, config = train_xgboost(data, rounds=5, threads=1)
    changed = copy.deepcopy(data)
    changed.sales[:, data.split.val_end :] = 50_000
    other, other_config = train_xgboost(changed, rounds=5, threads=1)
    assert trees.save_raw() == other.save_raw()
    assert config["best_round"] == other_config["best_round"]
    assert trees.num_boosted_rounds() == config["best_round"]
    origins = data.split.origins("test")
    before = predict_xgboost(trees, data, origins)
    path = tmp_path / "trees.json"
    trees.save_model(path)
    restored = xgb.Booster(params={"nthread": 1})
    restored.load_model(path)
    np.testing.assert_allclose(before, predict_xgboost(restored, data, origins))
    np.testing.assert_allclose(before[2:3], predict_xgboost(restored, data, origins, [2]))
    rows, by_date, metrics = backtest(DemandCNN(3), data, origins, torch.device("cpu"), before)
    assert metrics["xgboost_mae"] == pytest.approx((rows.xgboost - rows.actual).abs().mean())
    assert metrics["xgboost_mae"] == pytest.approx(by_date.xgboost_mae.mean())
    assert before.shape == (3, 4, 7)
    assert (before >= 0).all()
