# Configuration

There is no configuration service or credentials file. Training options are CLI
arguments; their values are saved in `preprocessing.json`.

| Option | Default | Purpose |
| --- | --- | --- |
| `--data-dir` | `data` | Directory containing the two M5 CSVs |
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

For separate experiments:

```bash
uv run --extra cpu python -m m5_forecast.train --epochs 5 --artifacts-dir artifacts/run-2
M5_ARTIFACTS_DIR=artifacts/run-2 uv run --extra cpu streamlit run app.py
```

All paths are relative to the working directory. Keep each run's saved files
together. A fixed seed helps repeat CPU runs; hardware and backend versions can
still affect numerical results.
