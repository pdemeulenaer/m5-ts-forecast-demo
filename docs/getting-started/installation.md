# Installation

## 1. Install dependencies

Use Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From the project root:

```bash
uv sync --extra cpu --group docs
```

The CPU extra installs CPU-only PyTorch. `uv.lock` records resolved versions;
the development group includes pytest and Ruff.

## 2. Download M5

Visit the [official Kaggle data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data),
sign in, and accept the competition rules if prompted. Download and extract:

```text
data/sales_train_evaluation.csv
data/calendar.csv
```

Also download `sell_prices.csv` for dollar-weighted WRMSSE. It is optional for
MAE and RMSSE. Use the evaluation sales file, which
contains the longer observed history. Raw CSVs are excluded from Git.

## 3. Train and view

```bash
uv run --extra cpu python -m m5_forecast.train --device cpu
uv run --extra cpu streamlit run app.py
```

Open <http://localhost:8501>. Select a product–store ID and a test forecast date
(the first day being predicted). Inspect the chart, seven-day total, daily table,
and aggregate test errors. Download the selected forecast as CSV if needed.

Training writes `artifacts/`. The app reads those saved files, so it can run
without the original dataset. Refresh it after a new training run. The app reloads
cached models when saved files or model-loading code change, and discards older
cached models missing forecast configuration. No retraining is needed after a code update.
The command trains both Conv1D and XGBoost. For an existing Conv1D run, add
`--xgboost-only` to train the comparison model using saved data.
To refresh competition-style metrics without training:

```bash
uv run --extra cpu python -m m5_forecast.metrics
```

## 4. Optional GPU

For an NVIDIA GPU with a CUDA 12.8-compatible driver:

```bash
uv sync --extra cuda
uv run --extra cuda python -m m5_forecast.train --device auto
uv run --extra cuda streamlit run app.py
```

Use one backend extra per command: `cpu` or `cuda`. `auto` falls back to CPU when
CUDA is unavailable; `--device cuda` requires it. The viewer always loads weights
on CPU. Backend setup follows [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

## 5. Docs and checks

```bash
make docs                 # http://127.0.0.1:8000
make docs PORT=8001
make docs-build
make test
make lint
```

Use `BACKEND=cuda` with Make targets when using the CUDA environment.
