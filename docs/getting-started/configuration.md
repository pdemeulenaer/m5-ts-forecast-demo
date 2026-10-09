# Configuration

There is no configuration service or credentials file. Training options are CLI
arguments; their values are saved in `preprocessing.json`.

## Original training options

These options apply to `python -m m5_forecast.train`, which selects checkpoints
by validation MAE.

| Option | Default | Purpose |
| --- | --- | --- |
| `--data-dir` | `data` | Sales/calendar CSVs; add prices for WRMSSE |
| `--artifacts-dir` | `artifacts` | Output directory (overwritten on another run) |
| `--num-series` | `100` | Number of regular sellers selected from training data |
| `--epochs` | `15` | Training passes; keep the best validation checkpoint |
| `--xgb-rounds` | `300` | Maximum tree boosting rounds; validation selects the best prefix |
| `--xgboost-only` | off | Add/update XGBoost using an existing saved Conv1D run |
| `--batch-size` | `256` | Windows per update |
| `--lr` | `0.001` | Adam learning rate |
| `--train-stride` | `7` | Days between training forecast dates |
| `--val-days` / `--test-days` | `56` / `56` | Latest held-out target periods |
| `--device` | `auto` | `cpu`, `cuda`, or automatic selection |
| `--threads` | `4` | CPU threads for PyTorch and XGBoost |
| `--seed` | `42` | NumPy and PyTorch random seed |

The lookback and horizon are fixed at 56 and 7 days. Validation needs at least
7 days; test needs at least 14 to allow multiple forecast dates. More training
windows can be created with `--train-stride 1`, at the cost of more CPU work.

XGBoost runs on CPU even when Conv1D uses CUDA. With `--xgboost-only`, series,
scales, split, training stride, and seed come from the saved run. Its settings
and validation result are saved separately in `xgboost_config.json`.

## Improvement options

Run `python -m m5_forecast.improve` after original training to compare candidates
and select models and adjustments by validation WRMSSE.

| Option | Default | Purpose |
| --- | --- | --- |
| `--artifacts-dir` | `artifacts` | Existing saved run; supplies series, scales, split, stride, and seed |
| `--data-dir` | `data` | All three M5 CSVs, including `sell_prices.csv` |
| `--output-dir` | existing run | Save separately; otherwise back up and update the input run |
| `--epochs` | `20` | Maximum residual CNN epochs |
| `--xgb-rounds` | `300` | Maximum rounds for each enhanced tree candidate |
| `--threads` | `4` | CPU threads for improvement training |

Improvement training runs on CPU. See [Improving the models](../operations/improvements.md)
for candidate features and losses. The [notebook](../operations/notebook.md) exposes
settings in its first configuration cell and saves both runs separately.

## Separate experiments

Save a run in its own directory and point Streamlit to it:

```bash
uv run --extra cpu python -m m5_forecast.train --epochs 5 --artifacts-dir artifacts/run-2
M5_ARTIFACTS_DIR=artifacts/run-2 uv run --extra cpu streamlit run app.py
```

All paths are relative to the working directory. Keep each run's saved files
together. A fixed seed helps repeat CPU runs; hardware and backend versions can
still affect numerical results.
