# Retail demand forecasting demo

A small PyTorch **Conv1D** model predicts seven daily sales values from the previous
56 days. One model learns across 100 regularly selling product–store series, with
series embeddings and calendar features. Streamlit compares it with a shared
**XGBoost** model and a baseline that repeats last week's sales, showing MAE for
each method on the selected forecast and across all test dates.
The overall dashboard also reports RMSSE and a competition-style WRMSSE for
the selected demo dataset.

## Run locally

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run commands from this directory:

```bash
uv sync --extra cpu
```

Download `sales_train_evaluation.csv` and `calendar.csv` from
[Kaggle's M5 Forecasting – Accuracy data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).
Sign in and accept the competition rules if prompted. Extract the two files into
`data/` (use **evaluation**, rather than `sales_train_validation.csv`).

```text
data/
├── sales_train_evaluation.csv
└── calendar.csv
```

```bash
uv run --extra cpu python -m m5_forecast.train --device cpu
uv run --extra cpu streamlit run app.py
```

Open <http://localhost:8501>, select a series and a test forecast date. The app loads
saved artifacts; it does not train or read the large CSVs on startup.

Training now fits both models. To add XGBoost to an existing saved Conv1D run:

```bash
uv run --extra cpu python -m m5_forecast.train --xgboost-only
```

This reuses the saved series, scales, split, and Conv1D checkpoint. XGBoost runs
on CPU and uses validation MAE for early stopping (`--xgb-rounds 300` by default).

For competition-style metrics, also download `sell_prices.csv` into `data/`.
Refresh an existing run's reports without retraining:

```bash
uv run --extra cpu python -m m5_forecast.metrics
```

Normal training updates these reports automatically. WRMSSE is calculated over
12 aggregation levels of the selected 100 series and our seven-day forecasts;
it is **not a full M5 leaderboard score**. Without prices, MAE and RMSSE still work.

To try the validated improvements (seasonal residual CNN, richer XGBoost features,
Tweedie loss, and total-demand adjustments):

```bash
uv run --extra cpu python -m m5_forecast.improve
```

This selects models on validation WRMSSE and updates the dashboard artifacts,
preserving the previous run in a timestamped backup. See
[measured improvements](docs/operations/improvements.md) for before/after results.

For an NVIDIA GPU with a driver supporting CUDA 12.8, use the `cuda` extra instead:

```bash
uv sync --extra cuda
uv run --extra cuda python -m m5_forecast.train --device auto
uv run --extra cuda streamlit run app.py
```

`auto` uses CUDA when available and otherwise uses CPU. The viewer runs on CPU even
for a GPU-trained checkpoint. The mutually exclusive extras follow the
[uv PyTorch setup guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Reproduce and experiment in a notebook

Open [notebooks/m5_experiments.ipynb](notebooks/m5_experiments.ipynb) to walk through
data preparation, the original models, validation-selected improvements, and test
comparisons. It requires the three M5 CSVs, including `sell_prices.csv`.

```bash
uv sync --extra cpu --group notebook
uv run --extra cpu --group notebook jupyter lab notebooks/m5_experiments.ipynb
```

In VS Code, select `.venv/bin/python` as the kernel. Settings and the original CNN
training loop are editable. Both runs are saved under `artifacts/notebook-01/`.
See the [notebook guide](docs/operations/notebook.md) for quick trials and how to
open notebook results in Streamlit.

## What the results mean

- The final 56 days are test targets; the preceding 56 days are validation targets.
  Each forecast uses only the 56 observations before its first predicted day.
- Selection ranks products by their fraction of nonzero training days, then training
  mean sales. Per-series sales scales are fitted only on training observations.
- Both models use the same weekly training dates by default. Validation chooses
  the Conv1D epoch and XGBoost tree count by MAE in original units. Testing uses
  eight weekly dates with seven targets each for all three methods.
- Test dates advance through time: previously observed test sales become history
  for later forecasts. Weights and preprocessing remain fixed.
- **MAE** is mean absolute error in units per day, averaged equally over all selected
  series, test forecast dates, and horizon days. Lower is better. A model that does
  not beat the baseline offers no demonstrated improvement on this backtest.
- **RMSSE** scales RMSE by observed daily sales changes after the first nonzero sale.
  **Demo WRMSSE** weights these errors by recent dollar sales across the hierarchy.
  Their rankings can differ from MAE; see the [evaluation guide](docs/operations/evaluation.md).

Training prints overall and per-date test MAE and saves everything to `artifacts/`:
weights, preprocessing metadata, selected data, training history, metrics, and test
forecasts. These files and the raw data are excluded from Git. Real M5 scores require
downloading the dataset and running training; no performance claim is assumed.

This demo estimates recorded sales. Stockouts can hide demand; procurement decisions
also need stock levels, lead times, and safety stock. Results for regular sellers
do not represent all M5 products. Official M5 scoring uses the full hierarchy and a
28-day horizon; this demo keeps the requested 56 → 7 forecasting task.

## Documentation and checks

The MkDocs site follows the `rag-demo` layout: Material theme, Getting Started,
Architecture, Operations, and a docstring-based Code Reference.

```bash
uv sync --extra cpu --group docs
uv run --extra cpu --group docs mkdocs serve
uv run --extra cpu --group docs mkdocs build --strict
uv run --extra cpu pytest
uv run --extra cpu ruff check .
```

See [installation](docs/getting-started/installation.md),
[the forecasting pipeline](docs/architecture/index.md), and
[evaluation](docs/operations/evaluation.md). For training options:

```bash
uv run --extra cpu python -m m5_forecast.train --help
```
