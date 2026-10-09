# Improving the models

## What the winning solution did

The winner used ensembles of LightGBM models pooled by store, store-category,
and store-department. It combined direct and recursive forecasts, used Tweedie
loss, and checked four recent 28-day validation windows for accuracy and stability.
These details are reported in the organizers'
[M5 results paper, section 4.2](https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf).
The supplied [Kaggle writeup](https://www.kaggle.com/competitions/m5-forecasting-accuracy/writeups/yeonjun-in-stu-1st-place-solution)
requires a browser to display its text; the recommendations below use the paper
and adapt the ideas to our existing models.

## Recommended experiments

Keep the same series and test periods when comparing an improvement. Tune on
validation only; reserve test results for the final comparison.

| Priority | XGBoost | Conv1D |
| --- | --- | --- |
| 1. Match the evaluation objective | Select rounds and settings by validation WRMSSE | Select checkpoints by validation WRMSSE; compare MAE loss with squared error |
| 2. Add demand features | Rolling means/std over 7, 14, 28, 56 days; zero fraction; time since last sale; horizon-aligned weekly lags | Predict a correction to last week's baseline; add rolling demand summaries |
| 3. Add explanatory variables | Past price, price change, item/store/category/department, event names | Price input channel, calendar event embeddings, separate item and store embeddings |
| 4. Handle intermittent demand | Try `reg:tweedie` with variance power between 1 and 2, or Poisson | Compare Poisson/Tweedie loss or a sale-occurrence plus positive-size model |
| 5. Combine forecasts | Validate a blend with Conv1D and the seasonal baseline | Small residual/dilated convolutions; average a few seeds if validation improves |

XGBoost's [objective documentation](https://xgboost.readthedocs.io/en/stable/parameter.html)
describes Tweedie and Poisson options. PyTorch provides
[PoissonNLLLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html).
Use nonnegative targets; evaluate forecasts in raw units. A weighted squared-error
loss can approximate the bottom-level scaled objective, but it does not reproduce
the whole hierarchical WRMSSE objective.

Compute every sales feature from observations strictly before the forecast date.
For a seven-day direct forecast, a matching weekday lag is available from the
previous week. Rolling statistics must end at the forecast origin, even for the
later horizon days. Future prices may be used only if they were scheduled and
known when the forecast was issued; otherwise use observed prices. Fit category
vocabularies on training data and reserve an unknown category for later events.

## Validation and coverage

Use several rolling validation cutoffs and inspect both mean and variation of
WRMSSE. Refit models and preprocessing within each fold. For this seven-day use
case, score separate seven-day origins; reproducing the competition requires
28-day forecasts made from a single cutoff, without observing intervening sales.

The current regular-seller subset underrepresents intermittent products. A later
experiment should use training-only stratification to include sparse sellers and
report errors by zero-sales fraction. Avoid store-specific models until each
group has enough training examples; shared models with group identity features
keep the current 100-series demo small.

## What is implemented now

The improvement command runs a fixed set of candidates on the existing selected
series and split:

```bash
uv run --extra cpu python -m m5_forecast.improve
```

It compares the original models with a seasonal residual CNN and two enhanced
XGBoost models (squared error and Tweedie). All choices use validation WRMSSE;
only the two selected methods are then evaluated on test. This command runs on
CPU. Set `--epochs`, `--xgb-rounds`, or `--threads` to control its runtime.

- **CNN:** begins with the mean of four matching weekdays, then learns positive
  multiplicative corrections. Training batches contain all series at the same
  forecast dates, so the loss includes the actual hierarchy's weighted errors.
- **XGBoost:** adds 7/14/28/56-day rolling means, standard deviations, zero-sales
  fractions, the training sales scale, and horizon-aligned weekday lags. Boosting
  rounds are selected by validation WRMSSE.
- **Both:** validation selects a fixed total-adjustment strength from
  `0, 0.25, 0.5, 0.75, 1`. This moves each predicted daily total toward last week's
  observed total, while retaining the model's predicted product shares.

The selected CNN used epoch 18. The selected tree model used enhanced features
and Tweedie loss. Both used adjustment strength 0.25. On the unchanged test set:

| Method | MAE before | MAE after | Demo WRMSSE before | Demo WRMSSE after |
| --- | ---: | ---: | ---: | ---: |
| Conv1D | 4.935 | 4.810 | 0.653 | **0.486** |
| XGBoost | 5.051 | 4.650 | 0.664 | **0.490** |
| Seasonal baseline | 5.917 | 5.917 | 0.618 | 0.618 |

These improvements apply to the 100-series, seven-day demo, not the full M5
competition. The already-inspected test period is useful for a before/after
comparison; a fresh holdout would provide stronger evidence of generalization.

The default command preserves existing files in a timestamped
`artifacts/before-improvements-*/` directory, then updates the active run. Refresh
Streamlit after completion. To keep experiments in a separate directory:

```bash
uv run --extra cpu python -m m5_forecast.improve --output-dir artifacts/experiment
M5_ARTIFACTS_DIR=artifacts/experiment uv run --extra cpu streamlit run app.py
```

`improvement_results.json` records every validation candidate, chosen adjustment,
and before/after scores. `residual_training_history.json` records the CNN epochs.
Saving preserves model flags and tree feature settings so app inference matches
the backtest. A normal `m5_forecast.train` run still trains the original models;
run `m5_forecast.improve` afterwards to apply this experiment workflow.

## Why the baseline originally won

MAE scores products individually. Hierarchical scoring also sums predictions
before measuring errors. For actual sales of 10 units each for two products,
predictions of 8 and 12 have MAE 2 but a correct total of 20. Predictions of 9 and
9 have a better MAE of 1 but an incorrect total of 18. Shared model bias can
therefore hurt aggregates even when product errors are smaller. Revenue weighting
changes the comparison further.

In the original run, total/category WRMSSE was about 0.40 for the baseline versus
0.60–0.62 for the learned models. The CNN also optimized MAE, and both checkpoints
were chosen by MAE. Hierarchical training and total adjustments directly address
that mismatch. No extra prices or future sales were used as prediction features.
