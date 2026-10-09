# Forecasting pipeline

## Data and selection

`data.py` reads the sales CSV in chunks. It ranks series by the fraction of training
days with positive sales, breaking ties by training mean sales and then ID. It
keeps 100 series by default and loads their full histories in a second pass.

Each selected series is divided by its mean training sales, floored at one. The
ordered ID list defines embedding indices; both IDs and scales are saved. No
validation or test sales affect selection or scaling.

Calendar inputs encode weekday and month as sine/cosine pairs, two event-presence
flags, and the selected store's state-specific SNAP flag. These deterministic
features require no fitted vocabulary. Future dates and scheduled events are
treated as known at forecast time.

## Chronological windows

With the usual 1,941 sales days and default splits:

| Period | Target days | Use |
| --- | --- | --- |
| Training | `d_1`–`d_1829` | Select series, fit scales, and learn weights |
| Validation | `d_1830`–`d_1885` | Choose the epoch with lowest unit MAE |
| Test | `d_1886`–`d_1941` | Final comparison at eight weekly forecast dates |

The first training forecast starts at `d_57`. At every forecast origin, the input
is the preceding 56 days and targets are the following seven days. Windows whose
targets cross a split boundary are excluded. Validation and test histories may
include earlier periods because those observations already exist at that date.

Test forecast dates advance by seven days. This is a rolling backtest: actual
sales from the previous test week become inputs to the next week, with no retraining
or scale updates. It is not one 56-day forecast made at the start of test.

## Model

The shared network uses two Conv1D layers (16 channels, kernel size 7) and ReLU.
It flattens the history representation, joins an eight-value product–store
embedding and the seven future days' calendar features, then predicts all seven
sales values through a small dense head. Softplus keeps predictions nonnegative.
There is no recursive feeding of predicted sales.

Training uses Adam and MAE in normalized sales units. Normalization keeps large
sellers from dominating training. Validation checkpoint selection and all reported
errors use original units, so they answer a different aggregation question.

Convolutions may mix positions within the history window; the whole window is
already observed. Neither the network nor its preprocessing uses future sales.

## XGBoost comparison

`xgboost_model.py` trains one shared tree model on the same series, normalized sales,
and chronological windows. Each training window produces seven rows: one per
horizon day. A row contains the 56 historical sales values, a one-hot series ID,
the horizon day (1–7), and that target date's known calendar features. Identity
uses the saved training ID order; no future sales enter these features.

The tree model uses squared-error training, histogram splits, and at most 300
boosting rounds. Validation MAE in original units selects the best round, with
early stopping after 20 rounds without improvement. Predictions are clipped at
zero and rescaled to units. Both models keep their training-only scales and use
the same test targets, though their feature representations and training losses
differ. XGBoost always trains and predicts on CPU.

Its [official Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
documents the training, custom validation metric, model slicing, and JSON persistence.

## Modules

```text
src/m5_forecast/
├── data.py        # selection, calendar features, chronological windows
├── model.py       # shared Conv1D network
├── adjustment.py  # past-only adjustment of forecast totals
├── improve.py     # validation-selected improvement experiments
├── xgboost_model.py # shared tree model and lag features
├── train.py       # CLI and training loop
├── evaluate.py    # prediction, baseline, MAE
├── metrics.py     # RMSSE, hierarchy/revenue weights, saved-run evaluation
└── artifacts.py   # save/load weights and preprocessing
app.py             # Streamlit viewer
tests/             # leakage, metrics, artifact, and UI checks
```
