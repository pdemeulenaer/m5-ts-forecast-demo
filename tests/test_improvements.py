"""Verify hierarchical optimization, forecast adjustments, and validation-only selection."""

import argparse
import copy

import numpy as np
import pytest
import torch
import xgboost as xgb

from m5_forecast.adjustment import adjust_totals
from m5_forecast.artifacts import load_artifacts, save_artifacts
from m5_forecast.data import SalesWindows, prepare_data
from m5_forecast.evaluate import predict
from m5_forecast.improve import run_improvements, select_adjustment
from m5_forecast.metrics import WRMSSEScorer, hierarchical_wrmsse, load_revenue
from m5_forecast.model import DemandCNN
from m5_forecast.train import run_training
from m5_forecast.xgboost_model import make_features, predict_xgboost, train_xgboost


def test_cached_scorer_matches_official_formula_and_ignores_future_history(m5_data):
    data = prepare_data(m5_data, num_series=3, val_days=28, test_days=28)
    origins = data.split.origins("validation")
    context = load_revenue(data, m5_data, origins)
    scorer = WRMSSEScorer(data, origins, *context)
    forecast = np.stack([data.sales[:, o : o + 7] + 2 for o in origins], axis=1)
    expected = [
        hierarchical_wrmsse(data, *context, forecast[:, i], int(o))[0]
        for i, o in enumerate(origins)
    ]
    assert scorer.score(forecast) == pytest.approx(np.mean(expected))
    changed = copy.deepcopy(data)
    changed.sales[:, data.split.val_end :] = 50_000
    other = WRMSSEScorer(changed, origins, *context)
    np.testing.assert_array_equal(scorer.scales, other.scales)
    np.testing.assert_array_equal(scorer.weights, other.weights)
    assert scorer.score(forecast) == other.score(forecast)


def test_total_adjustment_preserves_shares_and_uses_only_previous_week(m5_data):
    data = prepare_data(m5_data, num_series=3, val_days=28, test_days=28)
    origins = data.split.origins("test")[:1]
    forecast = np.full((3, 1, 7), 5.0)
    adjusted = adjust_totals(forecast, data, origins, 1)
    np.testing.assert_allclose(
        adjusted.sum(0), data.sales[:, origins[0] - 7 : origins[0]].sum(0)[None]
    )
    np.testing.assert_allclose(adjusted[0], adjusted[1])  # equal model shares remain equal
    changed = copy.deepcopy(data)
    changed.sales[:, origins[0] :] = 90_000
    np.testing.assert_array_equal(adjusted, adjust_totals(forecast, changed, origins, 1))
    assert adjust_totals(forecast, data, origins, 0) is forecast
    zero = adjust_totals(np.zeros_like(forecast), data, origins, 0.5)
    assert np.isfinite(zero).all()
    np.testing.assert_allclose(zero.sum(0), 0.5 * adjusted.sum(0))


def test_residual_cnn_starts_at_weekly_average_and_roundtrips(m5_data, tmp_path):
    data = prepare_data(m5_data, num_series=3, val_days=28, test_days=28)
    model = DemandCNN(3, seasonal_residual=True)
    origins = data.split.origins("test")
    past, calendar, series, _ = SalesWindows(data, origins)[0]
    with torch.inference_mode():
        output = model(past[None], calendar[None], series[None])
    torch.testing.assert_close(output[0], past[0, -28:].reshape(4, 7).mean(0))
    model.reconcile_alpha = 0.75
    before = predict(model, data, origins, torch.device("cpu"))
    save_artifacts(tmp_path, model, data, {}, {})
    restored, _, _ = load_artifacts(tmp_path)
    assert restored.seasonal_residual
    assert restored.reconcile_alpha == 0.75
    np.testing.assert_allclose(before, predict(restored, data, origins, torch.device("cpu")))


def test_enhanced_tree_features_and_adjusted_predictions_roundtrip(m5_data, tmp_path):
    data = prepare_data(m5_data, num_series=3, val_days=28, test_days=28)
    origins = data.split.origins("test")[:1]
    features = make_features(data, origins, enhanced=True)
    changed = copy.deepcopy(data)
    changed.sales[:, origins[0] :] = 80_000
    np.testing.assert_array_equal(features, make_features(changed, origins, enhanced=True))
    assert features.shape[1] == make_features(data, origins).shape[1] + 16
    val_origins = data.split.origins("validation")
    scorer = WRMSSEScorer(data, val_origins, *load_revenue(data, m5_data, val_origins))
    trees, config = train_xgboost(
        data, rounds=3, threads=1, enhanced=True, objective="reg:tweedie", scorer=scorer
    )
    assert config["selection_metric"] == "wrmsse"
    trees.set_attr(reconcile_alpha="0.75")
    path = tmp_path / "trees.json"
    trees.save_model(path)
    restored = xgb.Booster(params={"nthread": 1})
    restored.load_model(path)
    before = predict_xgboost(trees, data, origins)
    np.testing.assert_allclose(before, predict_xgboost(restored, data, origins))
    np.testing.assert_allclose(before[1:2], predict_xgboost(restored, data, origins, [1]))


def test_adjustment_is_selected_by_validation_score(m5_data):
    data = prepare_data(m5_data, num_series=3, val_days=28, test_days=28)
    origins = data.split.origins("validation")
    scorer = WRMSSEScorer(data, origins, *load_revenue(data, m5_data, origins))
    forecast = np.stack([data.sales[:, o : o + 7] * 0.5 for o in origins], axis=1)
    score, alpha = select_adjustment(forecast, data, origins, scorer)
    assert alpha == 1
    assert score == pytest.approx(0)


def test_improvement_run_selects_by_validation_and_preserves_source(m5_data, tmp_path):
    source, output = tmp_path / "source", tmp_path / "improved"
    args = argparse.Namespace(
        data_dir=m5_data,
        artifacts_dir=source,
        num_series=3,
        val_days=28,
        test_days=28,
        epochs=1,
        batch_size=32,
        train_stride=7,
        threads=1,
        lr=0.001,
        seed=42,
        device="cpu",
        xgb_rounds=2,
        xgboost_only=False,
    )
    run_training(args)
    checkpoint = (source / "model.pt").read_bytes()
    report = run_improvements(
        argparse.Namespace(
            artifacts_dir=source,
            output_dir=output,
            data_dir=m5_data,
            epochs=1,
            xgb_rounds=2,
            threads=1,
        )
    )
    assert (source / "model.pt").read_bytes() == checkpoint
    for prefix, label in [("cnn", "CNN"), ("xgboost", "XGBoost")]:
        candidates = [row for row in report["validation_candidates"] if label in row["candidate"]]
        assert report[f"{prefix}_validation_wrmsse"] == min(
            row["validation_wrmsse"] for row in candidates
        )
    assert (
        report["before_test_metrics"]["num_predictions"]
        == report["after_test_metrics"]["num_predictions"]
    )
    restored, data, metadata = load_artifacts(output)
    assert metadata["run_config"]["selection_metric"] == "validation_wrmsse"
    predictions = predict(restored, data, data.split.origins("test"), torch.device("cpu"))
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()
