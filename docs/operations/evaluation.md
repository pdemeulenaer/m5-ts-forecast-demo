# Evaluation and artifacts

## Read the comparison

The baseline predicts the next Monday from the previous Monday's sales, and so on:
it simply repeats the last seven observations. Conv1D, XGBoost, and the baseline are evaluated on
exactly the same series, dates, and targets.

MAE is the mean of `abs(prediction - actual)` across series, forecast dates, and
all seven horizon days, in original sales units. A score of 2 means the average
daily prediction is two units away from the observed value. Weekly target windows
do not overlap. If a custom held-out period is not a multiple of seven, trailing
days without a complete weekly window are not scored.

The training command prints per-date and aggregate MAE for all three methods;
Streamlit shows an overall comparison at the top and selected-window MAE/RMSSE
below it. The chart, daily
table, and downloadable CSV include both learned forecasts and the baseline.
The interface's selected-forecast score covers only one series and one date, so
it can differ considerably from the aggregate score. Lower MAE is better; compare
against the baseline before attributing value to the neural model. Test data must
not guide parameter tuning. No real M5 score is supplied before training on the CSVs.

## Competition-style metrics

For each series, RMSSE is `sqrt(forecast MSE / historical mean squared daily change)`.
The historical calculation starts at the first nonzero sale. WRMSSE aggregates
sales and forecasts into twelve levels, scores each aggregate, weights by the
preceding 28 days' dollar sales within each level, then averages the levels.
See the [official competitors' guide, pages 6–9](https://storage.googleapis.com/kaggle-forum-message-attachments/772349/15032/M5-Competitors-Guide-Final-10-March-2020.pdf).

Our **demo WRMSSE** uses the selected series and seven-day horizon. Partial store,
category, and total aggregates contain only these products. It is not comparable
to the full M5 leaderboard, which requires all 30,490 bottom series, their 42,840
aggregates, and a single 28-day forecast horizon. Chaining four weekly forecasts
using newly observed sales would not reproduce that task.

Scales and weights are computed separately at each origin from earlier
observations; no future sales or prices enter them. These scoring statistics do
not update model preprocessing. Overall mean RMSSE averages item-level scores
over series and dates; overall demo WRMSSE averages date-level hierarchy scores.
Neither is RMSSE of all errors pooled together. Undefined scales are shown as
missing, without epsilon adjustments or silently dropping positive-weight series.

Dollar weights require `sell_prices.csv`. Without it, MAE and RMSSE remain
available and WRMSSE is shown as missing. The metric refresh command preserves
model weights and predictions:

```bash
uv run --extra cpu python -m m5_forecast.metrics --data-dir data --artifacts-dir artifacts
```

## Current M5 results

To reproduce both runs step by step, use the [experiment notebook](notebook.md).

The active saved run evaluates 100 selected product–store series across eight
weekly test dates: 5,600 daily predictions with a seven-day horizon. These scores
come from `artifacts/metrics.json` and match the dashboard's overall comparison.

| Method | Test MAE (units/day) | Mean RMSSE | Demo WRMSSE |
| --- | ---: | ---: | ---: |
| Conv1D | 4.810 | 0.673 | **0.486** |
| XGBoost | **4.650** | **0.643** | 0.490 |
| Previous-week baseline | 5.917 | 0.842 | 0.618 |

Both learned models now beat the baseline on all three metrics. XGBoost has the
lowest item-level MAE and mean RMSSE; Conv1D has the lowest demo WRMSSE. Aggregate
errors and revenue weights can rank the methods differently from item-level MAE.

The CNN learns corrections to a four-week weekday average and trains with a
hierarchical WRMSSE loss; its saved checkpoint is epoch 18. XGBoost uses enhanced
rolling-demand features and Tweedie loss, with 33 boosting rounds. Both apply a
0.25 total-adjustment strength toward the previous week's daily totals. Settings
were selected using validation WRMSSE. See [Improving the models](improvements.md)
for the training command and implementation details.

### Before improvements

The original models used validation MAE to select CNN epoch 5 and XGBoost round
85. On the same series and test dates, they produced:

| Method | Test MAE (units/day) | Mean RMSSE | Demo WRMSSE |
| --- | ---: | ---: | ---: |
| Original Conv1D | 4.935 | 0.675 | 0.653 |
| Original XGBoost | 5.051 | 0.683 | 0.664 |
| Previous-week baseline | 5.917 | 0.842 | 0.618 |

The baseline originally won on demo WRMSSE despite higher MAE. Hierarchical
training, richer tree features, and validation-selected total adjustments improved
the learned models' scores. The comparison uses an already-inspected test period;
a fresh holdout would provide stronger evidence of generalization. Retraining may
change the results; the dashboard always reads the current saved run.

## Saved files

| File in `artifacts/` | Contents |
| --- | --- |
| `model.pt` | Best-validation state dict, saved as CPU tensors |
| `xgboost.json` | XGBoost trees through the best validation round |
| `xgboost_config.json` | Tree parameters, seed, training stride, and validation result |
| `preprocessing.json` | Ordered series IDs, scales, features, splits, run options, metrics |
| `demo_data.npz` | Selected raw sales, calendar arrays, and dates |
| `test_forecasts.csv` | Each actual value and `cnn`, `xgboost`, `baseline` predictions |
| `mae_by_date.csv` | MAE, mean RMSSE, and demo WRMSSE for each forecast date |
| `metrics.json` | Overall scores, evaluation counts, and explicit scoring scope |
| `metrics_by_series.csv` | Each series' MAE and mean RMSSE across test dates |
| `wrmsse_by_level.csv` | Each method's score at each hierarchy level and test date |
| `training_history.json` | Active CNN validation history: MAE originally, WRMSSE after improvements |
| `improvement_results.json` | Validation candidate selection and before/after scores |
| `residual_training_history.json` | Hierarchical residual CNN validation scores by epoch |

Report columns use method prefixes `cnn`, `xgboost`, and `baseline`, followed by
`_mae`, `_rmsse`, or `_wrmsse`. WRMSSE is a dataset metric; the selected series'
unweighted scaled error is RMSSE.

The app caches saved weights, trees, and prepared data, then runs CPU inference for the
selected window. It never invokes training. Cache keys include file modification
times for artifacts and model-loading code; older cached models missing forecast
configuration are discarded. A refreshed page can pick up a new completed run.
Stop the viewer while overwriting a run to avoid loading partially written artifacts. Use separate
artifact directories to keep experiments.

For a saved run created before XGBoost was added, run:

```bash
uv run --extra cpu python -m m5_forecast.train --xgboost-only
```

This fits XGBoost from saved data and refreshes the comparison reports without
changing the Conv1D checkpoint or its training history. Refresh the dashboard
after it completes. A new normal training run fits both models.

## Checks and limitations

```bash
uv run --extra cpu pytest
uv run --extra cpu ruff check .
uv run --extra cpu --group docs mkdocs build --strict
```

Tests use small synthetic M5-shaped CSVs, so no dataset download is needed. They
check training-only selection/scaling, target boundaries, future-input isolation,
the baseline, both models' unit MAE, known-answer RMSSE/WRMSSE calculations,
past-only price weights, CPU save/load, and the app's saved-artifact flow.
They also check that changing test sales cannot change XGBoost training. Synthetic
checks validate plumbing, not accuracy on Walmart data. With the `notebook`
dependency group installed, the suite also runs the notebook's cells on synthetic
data and verifies both exported runs.

The model learns sales rather than unconstrained demand. Stockouts, promotions,
prices, and changing assortment are not modeled. Only regular sellers are selected;
performance does not generalize automatically to intermittent products. The
seven-day forecast total informs planning but is not an order quantity: inventory,
lead times, and service-level decisions belong in a separate planning step. This
demo's scores cover regular sellers and a shorter horizon than the competition.
See [Improving the models](improvements.md) for the next experiments.
