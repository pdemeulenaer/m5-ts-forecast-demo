"""Exercise real CPU training and Streamlit against saved synthetic artifacts."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import m5_forecast.artifacts as artifacts_module
import m5_forecast.model as model_module
from m5_forecast.artifacts import load_artifacts
from m5_forecast.metrics import rmsse


def test_training_cli_and_saved_artifact_ui(m5_data, tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "m5_forecast.train",
            "--data-dir",
            str(m5_data),
            "--artifacts-dir",
            str(artifacts),
            "--num-series",
            "3",
            "--epochs",
            "2",
            "--val-days",
            "28",
            "--test-days",
            "28",
            "--device",
            "cpu",
            "--threads",
            "1",
            "--xgb-rounds",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Test MAE: cnn=" in result.stdout
    metrics = json.loads((artifacts / "metrics.json").read_text())
    assert metrics["num_forecast_dates"] == 4
    assert metrics["num_predictions"] == 84
    history = json.loads((artifacts / "training_history.json").read_text())
    metadata = json.loads((artifacts / "preprocessing.json").read_text())
    assert metadata["run_config"]["best_validation_mae"] == min(
        row["validation_mae"] for row in history
    )
    checkpoint = (artifacts / "model.pt").read_bytes()
    # Add/update the comparison using only saved data, without changing the CNN checkpoint.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "m5_forecast.train",
            "--xgboost-only",
            "--artifacts-dir",
            str(artifacts),
            "--xgb-rounds",
            "5",
            "--threads",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (artifacts / "model.pt").read_bytes() == checkpoint
    m5_data.rename(tmp_path / "hidden_raw_data")  # the viewer needs saved files only
    metrics = json.loads((artifacts / "metrics.json").read_text())
    monkeypatch.setenv("M5_ARTIFACTS_DIR", str(artifacts))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert len(app.sidebar.selectbox) == 2
    assert len(app.sidebar.selectbox[0].options) == 3
    assert len(app.sidebar.selectbox[1].options) == 4
    app.sidebar.selectbox[0].select_index(2)
    app.sidebar.selectbox[1].select_index(3)
    app.run()
    assert not app.exception
    overall = app.dataframe[0].value
    daily = app.dataframe[1].value
    assert len(daily) == 7
    rows = pd.read_csv(artifacts / "test_forecasts.csv")
    selected = rows[
        (rows.series_id == app.sidebar.selectbox[0].value)
        & (rows.forecast_date == app.sidebar.selectbox[1].value)
    ]
    pd.testing.assert_series_equal(
        daily["Conv1D"].reset_index(drop=True),
        selected["cnn"].round(2).reset_index(drop=True),
        check_names=False,
        check_dtype=False,
        atol=1e-5,
        rtol=1e-5,
    )
    assert float(app.metric[0].value) == pytest.approx(
        (selected.cnn - selected.actual).abs().mean(), abs=0.005
    )
    pd.testing.assert_series_equal(
        daily["XGBoost"].reset_index(drop=True),
        selected["xgboost"].round(2).reset_index(drop=True),
        check_names=False,
        check_dtype=False,
        atol=1e-5,
        rtol=1e-5,
    )
    assert float(app.metric[1].value) == pytest.approx(
        (selected.xgboost - selected.actual).abs().mean(), abs=0.005
    )
    assert len(app.metric) == 6
    for index, key in enumerate(["cnn", "xgboost", "baseline"]):
        assert overall.iloc[index]["MAE (units/day)"] == pytest.approx(
            metrics[f"{key}_mae"], abs=0.0005
        )
        assert overall.iloc[index]["Demo WRMSSE"] == pytest.approx(
            metrics[f"{key}_wrmsse"], abs=0.0005
        )
    _, data, _ = load_artifacts(artifacts)
    series = data.series_ids.index(app.sidebar.selectbox[0].value)
    origin = int((data.dates == app.sidebar.selectbox[1].value).nonzero()[0][0])
    assert float(app.metric[4].value) == pytest.approx(
        rmsse(selected.actual.to_numpy(), selected.xgboost.to_numpy(), data.sales[series, :origin]),
        abs=0.0005,
    )
    assert "XGBoost MAE" in app.dataframe[2].value.columns

    # Simulate a running Streamlit session retaining a pre-upgrade model instance.
    def outdated_constructor(num_series):
        raise AssertionError("The loader reused an outdated imported model class.")

    monkeypatch.setattr(model_module, "DemandCNN", outdated_constructor)
    monkeypatch.setattr(artifacts_module, "DemandCNN", outdated_constructor)
    source = (Path(__file__).parents[1] / "app.py").read_text()
    source += """
if 'stale_model_id' not in st.session_state:
    st.session_state.stale_model_id = id(model)
    del model.seasonal_residual
    del model.reconcile_alpha
else:
    st.session_state.model_reloaded = id(model) != st.session_state.stale_model_id
    st.session_state.model_config = (model.seasonal_residual, model.reconcile_alpha)
"""
    stale_app = AppTest.from_string(source, default_timeout=20).run()
    assert not stale_app.exception
    before = stale_app.dataframe[1].value.copy()
    stale_app.run()
    assert not stale_app.exception
    assert stale_app.session_state.model_reloaded
    assert stale_app.session_state.model_config == (False, 0)
    pd.testing.assert_frame_equal(before, stale_app.dataframe[1].value)


def test_app_without_artifacts_shows_training_command(tmp_path, monkeypatch):
    monkeypatch.setenv("M5_ARTIFACTS_DIR", str(tmp_path / "missing"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert app.info
    assert "m5_forecast.train" in app.code[0].value
    assert not app.sidebar.selectbox
