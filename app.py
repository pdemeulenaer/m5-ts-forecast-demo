"""Streamlit viewer for a saved M5 model and its test backtest."""

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import xgboost as xgb

import m5_forecast.artifacts as artifacts_module
import m5_forecast.model as model_module
from m5_forecast.data import HISTORY, HORIZON
from m5_forecast.evaluate import predict
from m5_forecast.metrics import rmsse
from m5_forecast.xgboost_model import predict_xgboost

st.set_page_config(page_title="Retail demand forecast", layout="wide")
st.title("Retail demand forecast")
st.write("Use the previous 56 days to forecast the next 7 days for one Walmart product–store.")

ARTIFACTS = Path(os.environ.get("M5_ARTIFACTS_DIR", "artifacts"))


def cached_model_is_current(resources: tuple) -> bool:
    """Discard models cached before forecast configuration was introduced."""
    model = resources[0]
    return hasattr(model, "seasonal_residual") and hasattr(model, "reconcile_alpha")


@st.cache_resource(validate=cached_model_is_current)
def load_demo(directory: str, version: tuple):
    """Cache saved CPU models until artifacts or model-loading code change."""
    torch.set_num_threads(4)
    # Streamlit reruns app.py, but imported modules can still contain older classes.
    importlib.reload(model_module)
    importlib.reload(artifacts_module)
    model, data, metadata = artifacts_module.load_artifacts(Path(directory))
    trees = xgb.Booster(params={"nthread": 4, "device": "cpu"})
    trees.load_model(Path(directory) / "xgboost.json")
    return model, trees, data, metadata


required = [
    ARTIFACTS / name
    for name in (
        "model.pt",
        "xgboost.json",
        "preprocessing.json",
        "demo_data.npz",
        "mae_by_date.csv",
    )
]
if not all(path.is_file() for path in required):
    st.info(
        "Train both models first, then refresh this page. "
        "For an existing Conv1D run, add --xgboost-only to the command below."
    )
    st.code("uv run --extra cpu python -m m5_forecast.train --device cpu", language="bash")
    st.stop()

try:
    source_files = [Path(model_module.__file__), Path(artifacts_module.__file__)]
    version = tuple(path.stat().st_mtime_ns for path in required + source_files)
    model, trees, data, metadata = load_demo(str(ARTIFACTS.resolve()), version)
except (ValueError, KeyError, RuntimeError, OSError) as error:
    st.error(f"Could not load artifacts: {error}")
    st.stop()

metrics = metadata["test_metrics"]
labels = [("Conv1D", "cnn"), ("XGBoost", "xgboost"), ("Previous week", "baseline")]
st.subheader(f"Overall comparison — all {metrics['num_series']} demo series")
st.caption(
    f"All {metrics['num_forecast_dates']} test forecast dates and all 7 horizon days. "
    "This covers the selected demo dataset, not all 30,490 M5 product–store series."
)
if metadata["run_config"].get("selected_cnn"):
    st.caption(
        f"Selected on validation WRMSSE: {metadata['run_config']['selected_cnn']}; "
        f"{metadata['run_config']['selected_xgboost']}."
    )
comparison = pd.DataFrame(
    [
        {
            "Method": name,
            "MAE (units/day)": metrics[f"{key}_mae"],
            "Mean RMSSE": metrics.get(f"{key}_rmsse"),
            "Demo WRMSSE": metrics.get(f"{key}_wrmsse"),
        }
        for name, key in labels
    ]
)
st.dataframe(comparison.round(3), hide_index=True)
st.caption(
    "Lower is better. Mean RMSSE averages scaled errors of individual series. "
    "Demo WRMSSE scores 12 aggregation levels with dollar-sales weights, averaged over test dates. "
    "Our 100-series, 7-day score is not comparable to the 28-day full M5 leaderboard."
)
if "wrmsse_status" not in metrics:
    st.info("Refresh scaled metrics with: uv run --extra cpu python -m m5_forecast.metrics")
elif metrics["wrmsse_status"] == "sell_prices.csv required":
    st.info(
        "WRMSSE needs sell_prices.csv. MAE and RMSSE remain available. "
        "Add the file to data/ and run: uv run --extra cpu python -m m5_forecast.metrics"
    )
elif metrics["wrmsse_status"] != "available":
    st.info("WRMSSE is undefined for constant history with positive weight, or zero total revenue.")
st.divider()

origins = data.split.origins("test")
series_id = st.sidebar.selectbox("Product–store series", data.series_ids)
forecast_date = st.sidebar.selectbox(
    "Test forecast date (first predicted day)", data.dates[origins]
)
series_index = data.series_ids.index(series_id)
origin = int(origins[np.flatnonzero(data.dates[origins] == forecast_date)[0]])
st.caption(
    f"History ends {data.dates[origin - 1]}. Forecast covers "
    f"{forecast_date} through {data.dates[origin + HORIZON - 1]}."
)

forecast = predict(model, data, np.asarray([origin]), torch.device("cpu"))[series_index, 0]
actual = data.sales[series_index, origin : origin + HORIZON]
baseline = data.sales[series_index, origin - HORIZON : origin]
tree_forecast = predict_xgboost(trees, data, np.asarray([origin]), [series_index])[0, 0]
methods = {"Conv1D": forecast, "XGBoost": tree_forecast, "Previous week": baseline}

st.subheader("This forecast")
for column, (name, values) in zip(st.columns(3), methods.items()):
    column.metric(f"{name} MAE (units/day)", f"{np.abs(values - actual).mean():.2f}")
for column, (name, values) in zip(st.columns(3), methods.items()):
    score = rmsse(actual, values, data.sales[series_index, :origin])
    column.metric(
        f"{name} RMSSE",
        f"{score:.3f}" if np.isfinite(score) else "Undefined",
        help="RMSE scaled by daily changes in observed history after the first sale. "
        "Undefined for constant history; lower is better.",
    )
st.caption(
    "7-day forecast totals: "
    + "; ".join(f"{name}: {values.sum():.1f} units" for name, values in methods.items())
)

chart = pd.DataFrame(index=pd.to_datetime(data.dates[origin - HISTORY : origin + HORIZON]))
chart.index.name = "Date"
chart["Recent history"] = np.r_[
    data.sales[series_index, origin - HISTORY : origin], np.full(HORIZON, np.nan)
]
for label, values in [
    ("Actual future sales", actual),
    ("Conv1D forecast", forecast),
    ("XGBoost forecast", tree_forecast),
    ("Previous-week baseline", baseline),
]:
    chart[label] = np.r_[np.full(HISTORY, np.nan), values]
st.line_chart(chart, x_label="Date", y_label="Sales (units)")

daily = pd.DataFrame(
    {
        "Date": data.dates[origin : origin + HORIZON],
        "Actual": actual,
        **methods,
    }
)
st.dataframe(daily.round(2), hide_index=True)
st.download_button(
    "Download this forecast",
    daily.to_csv(index=False),
    file_name=f"forecast-{series_id}-{forecast_date}.csv",
    mime="text/csv",
)

st.subheader("Overall comparison by forecast date")
st.caption(
    f"Average across {metrics['num_series']} series, "
    f"{metrics['num_forecast_dates']} weekly forecast dates, and all 7 horizon days. "
    "Later test forecasts use sales observed before their forecast date; the model stays fixed."
)
by_date = pd.read_csv(ARTIFACTS / "mae_by_date.csv").rename(
    columns={
        "forecast_date": "Forecast date",
        "cnn_mae": "Conv1D MAE",
        "xgboost_mae": "XGBoost MAE",
        "baseline_mae": "Previous-week MAE",
        "cnn_rmsse": "Conv1D RMSSE",
        "xgboost_rmsse": "XGBoost RMSSE",
        "baseline_rmsse": "Previous-week RMSSE",
        "cnn_wrmsse": "Conv1D demo WRMSSE",
        "xgboost_wrmsse": "XGBoost demo WRMSSE",
        "baseline_wrmsse": "Previous-week demo WRMSSE",
    }
)
st.dataframe(by_date.round(3), hide_index=True)
st.caption(
    "Lower MAE is better. Forecast totals can inform procurement; order quantities also "
    "depend on stock, lead times, and safety stock. "
    "Recorded sales may understate demand during stockouts."
)
