# Notebook experiments

Open `notebooks/m5_experiments.ipynb` to reproduce the original and improved runs
one stage at a time. The notebook uses the demo's data, models, features, and metrics;
the original CNN training loop is visible so its loss and optimizer are easy to edit.

## Start the notebook

Download all three M5 files into `data/`: `sales_train_evaluation.csv`, `calendar.csv`,
and `sell_prices.csv`. Prices are required for validation WRMSSE in this workflow.
See [Installation](../getting-started/installation.md) for the download link.

From the repository root:

```bash
uv sync --extra cpu --group notebook
uv run --extra cpu --group notebook jupyter lab notebooks/m5_experiments.ipynb
```

In VS Code, open the notebook and select `.venv/bin/python` as the kernel. Run cells
from top to bottom. Jupyter can start in the repository root or `notebooks/`; paths
are resolved against the repository root.

## Walkthrough

| Stage | What you can inspect or change |
| --- | --- |
| Settings and data | Seed, series count, split, training stride, epochs, and CPU threads |
| Windows and baseline | Input shapes, training history, calendar features, and baseline validation scores |
| Original Conv1D | Model layers, MAE training loop, optimizer, and checkpoint history |
| Original XGBoost | Lag features and validation MAE selection |
| Improved models | Seasonal residual CNN, hierarchical loss, richer tree features, and Tweedie loss |
| Validation selection | Candidate scores and daily-total adjustment strengths |
| Test evaluation | Before/after MAE, RMSSE, WRMSSE, per-date scores, and one forecast plot |
| Save | Complete original and improved artifact directories for Streamlit |

Defaults match the documented CPU experiment: 100 series, 15 original CNN epochs,
20 residual CNN epochs, and up to 300 tree rounds. For a quick trial, set
`NUM_SERIES = 10`, both epoch counts to `2`, and `XGB_ROUNDS = 10`.
Quick trials are functional checks rather than reproductions of the reported scores.

The notebook trains from local CSVs; it does not substitute existing checkpoints.
Exact scores can vary with hardware, package versions, and modified settings.
Compare with the [current and original results](evaluation.md#current-m5-results).

## Keep and view experiments

Set a new `RUN_DIR` for each experiment. The default writes:

```text
artifacts/notebook-01/
├── original/                 # Original model weights, settings, and reports
├── improved/                 # Validation-selected weights, settings, and reports
├── validation_candidates.csv
├── comparison.csv
└── comparison_by_date.csv
```

Open either saved run in the dashboard:

```bash
M5_ARTIFACTS_DIR=artifacts/notebook-01/improved uv run --extra cpu streamlit run app.py
```

Changing a training cell requires rerunning the selection, evaluation, and saving
cells after it. Changing selection, scaling, or splits requires rebuilding every
downstream stage. Restart the kernel after editing imported Python modules.

Choose candidates using validation results. The notebook computes test scores only
after model and adjustment choices are fixed. Once test results have informed new
experiments, use a fresh holdout for stronger evidence of generalization.

To run the notebook's small synthetic integration test:

```bash
uv run --extra cpu --group notebook pytest tests/test_notebook.py
```
