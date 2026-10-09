"""Protect the temporal split, preprocessing, and reported unit metrics."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from m5_forecast.artifacts import load_artifacts, save_artifacts
from m5_forecast.data import HISTORY, HORIZON, SalesWindows, Split, prepare_data
from m5_forecast.evaluate import backtest, predict, seasonal_baseline
from m5_forecast.model import DemandCNN


def test_selection_and_scaling_ignore_held_out_sales(m5_data):
    before = prepare_data(m5_data, num_series=3, val_days=28, test_days=28)
    path = m5_data / "sales_train_evaluation.csv"
    frame = pd.read_csv(path)
    future = [f"d_{i + 1}" for i in range(before.split.train_end, before.split.total_days)]
    frame[future] = 100_000  # even the intermittent fourth item sells heavily later
    frame.to_csv(path, index=False)
    after = prepare_data(m5_data, num_series=3, val_days=28, test_days=28)
    assert before.series_ids == after.series_ids == frame["id"].tolist()[:3]
    np.testing.assert_array_equal(before.scales, after.scales)
    np.testing.assert_allclose(before.scales, before.sales[:, : before.split.train_end].mean(1))


def test_targets_stay_inside_each_split():
    split = Split.create(1941)
    for period, start, end in [
        ("train", HISTORY, split.train_end),
        ("validation", split.train_end, split.val_end),
        ("test", split.val_end, split.total_days),
    ]:
        origins = split.origins(period)
        assert origins.min() >= start
        assert origins.max() + HORIZON <= end
        assert (origins - HISTORY >= 0).all()
    assert len(split.origins("test")) == 8
    assert split.train_end == 1829


def test_future_sales_cannot_change_forecast_inputs(m5_data):
    data = prepare_data(m5_data, num_series=3, val_days=28, test_days=28)
    origin = data.split.origins("test")[1]
    changed = copy.deepcopy(data)
    changed.sales[:, origin:] = 9000
    past, future_calendar, series, target = SalesWindows(data, np.array([origin]))[0]
    new_past, new_calendar, _, new_target = SalesWindows(changed, np.array([origin]))[0]
    torch.testing.assert_close(past, new_past)
    torch.testing.assert_close(future_calendar, new_calendar)
    assert not torch.equal(target, new_target)
    assert past.shape == (8, 56)
    assert future_calendar.shape == (7, 7)
    torch.testing.assert_close(
        past[0], torch.from_numpy(data.sales[0, origin - HISTORY : origin] / data.scales[0])
    )
    model = DemandCNN(3).eval()
    with torch.inference_mode():
        torch.testing.assert_close(
            model(past[None], future_calendar[None], series[None]),
            model(new_past[None], new_calendar[None], series[None]),
        )


def test_snap_uses_the_store_state(m5_data):
    data = prepare_data(m5_data, num_series=3, val_days=28, test_days=28)
    calendar = pd.read_csv(m5_data / "calendar.csv")
    for series, state in enumerate(["CA", "TX", "WI"]):
        np.testing.assert_array_equal(data.calendar[series, :, -1], calendar[f"snap_{state}"])


def test_baseline_repeats_previous_seven_observations():
    sales = np.arange(40).reshape(2, 20)
    origins = np.array([7, 13])
    baseline = seasonal_baseline(sales, origins)
    np.testing.assert_array_equal(baseline[:, 0], sales[:, :7])
    np.testing.assert_array_equal(baseline[:, 1], sales[:, 6:13])


def test_backtest_reports_mae_in_raw_units(m5_data):
    data = prepare_data(m5_data, num_series=3, val_days=28, test_days=28)
    model = DemandCNN(3)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    # Softplus(0) = log(2), then each series' training mean restores raw units.
    origins = data.split.origins("test")
    rows, by_date, summary = backtest(model, data, origins, torch.device("cpu"))
    expected = np.log(2) * data.scales
    for i, series_id in enumerate(data.series_ids):
        np.testing.assert_allclose(
            rows.loc[rows.series_id == series_id, "cnn"], expected[i], rtol=1e-6
        )
    assert summary["cnn_mae"] == pytest.approx((rows.cnn - rows.actual).abs().mean())
    assert summary["baseline_mae"] == pytest.approx(0.0)
    assert summary["num_predictions"] == 3 * 4 * 7
    assert len(by_date) == 4
    assert summary["cnn_mae"] == pytest.approx(by_date.cnn_mae.mean())


def test_saved_model_and_preprocessing_reproduce_forecasts(m5_data, tmp_path):
    data = prepare_data(m5_data, num_series=3, val_days=28, test_days=28)
    model = DemandCNN(3)
    origins = data.split.origins("test")
    before = predict(model, data, origins, torch.device("cpu"))
    directory = tmp_path / "artifacts"
    save_artifacts(directory, model, data, {"seed": 42}, {"cnn_mae": 1.0})
    restored_model, restored_data, metadata = load_artifacts(directory)
    after = predict(restored_model, restored_data, origins, torch.device("cpu"))
    np.testing.assert_allclose(before, after)
    assert metadata["series_ids"] == data.series_ids
    np.testing.assert_array_equal(restored_data.scales, data.scales)
    np.testing.assert_array_equal(restored_data.calendar, data.calendar)
    assert (after >= 0).all()


@pytest.mark.parametrize("days,val,test", [(100, 56, 56), (200, 6, 56), (200, 56, 7)])
def test_invalid_splits_fail_clearly(days, val, test):
    with pytest.raises(ValueError):
        Split.create(days, val, test)


def test_missing_csv_explains_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download the M5 CSVs"):
        prepare_data(tmp_path)
