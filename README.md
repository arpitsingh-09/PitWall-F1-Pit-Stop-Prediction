# PitWall: F1 Pit Stop Prediction

Predicting whether a Formula 1 driver will pit on the next lap, using XGBoost.

Built for the Kaggle [Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5) competition. Submissions are scored on ROC AUC.

## Overview

Pit stop timing is one of the most important strategic calls in Formula 1. This project treats it as a binary classification problem: given the state of a driver on a given lap (tyre age, compound, position, lap-time trend, and so on), what is the probability that they pit at the end of that lap?

- **Target:** `PitNextLap` (1 = pits next lap, 0 = does not)
- **Challenge:** the classes are heavily imbalanced, since most laps are not followed by a pit stop
- **Model:** `XGBClassifier` with class weighting

## Dataset

Each row is one driver on one lap of a race. Data comes from the Kaggle competition (`train.csv`, `test.csv`, `sample_submission.csv`).

| Feature | Description |
| --- | --- |
| `Driver` | Driver identifier |
| `Race` | Grand Prix name |
| `Year` | Season |
| `Compound` | Tyre compound (SOFT, MEDIUM, HARD, INTERMEDIATE, WET) |
| `TyreLife` | Laps completed on the current set of tyres |
| `Stint` | Current stint number |
| `LapNumber` | Lap of the race |
| `Position` | Current race position |
| `Position_Change` | Change in position |
| `LapTime_Delta` | Lap-time change, an indicator of tyre wear |
| `Cumulative_Degradation` | Accumulated degradation |
| `RaceProgress` | Fraction of the race completed (dropped, see below) |
| `PitStop` | Whether the driver pitted on the current lap |
| `PitNextLap` | **Target** |

The training data has no missing values.

## Approach

1. **Exploratory analysis:** value counts, summary statistics, target distribution by driver, compound and year, and a correlation heatmap.
2. **Feature engineering:**
   - Drop `RaceProgress` (redundant with `LapNumber`).
   - Add `laps_since_pit`, a running count of laps since the last pit stop within each Driver/Race group.
   - Ordinal-encode `Compound` in the order SOFT, MEDIUM, HARD, INTERMEDIATE, WET.
   - Target-encode `Driver` and `Race`.
   - Drop `PitStop` after deriving `laps_since_pit`.
3. **Modeling:**
   - 80/20 stratified train/validation split.
   - `XGBClassifier` with 500 estimators, learning rate 0.05, and `scale_pos_weight` set to the negative/positive class ratio.
   - Evaluated with ROC AUC and a classification report.
4. **Submission:** apply the same preprocessing to the test set with the fitted encoders, retrain on all training data, and write predicted probabilities to `submission.csv`.

## Results

| Metric | Score |
| --- | --- |
| Validation AUC | _fill in from notebook output_ |
| Public leaderboard AUC | _fill in after submitting_ |

## Getting started

### Requirements

```
numpy
pandas
scikit-learn >= 1.3   # TargetEncoder
xgboost
matplotlib
seaborn
```

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

### Running

1. Download the competition data from Kaggle and place `train.csv`, `test.csv` and `sample_submission.csv` in the input directory.
2. Update the file paths at the top of the notebook (they currently point to `/kaggle/input/competitions/playground-series-s6e5/`).
3. Run `xgb-prediction-f1-pit-stops.ipynb` top to bottom.
4. The notebook writes `submission.csv` with columns `id` and `PitNextLap`.

## Project structure

```
.
├── xgb-prediction-f1-pit-stops.ipynb   # EDA, features, model, submission
├── submission.csv                      # generated output
└── README.md
```

## Known issues and next steps

- Fix the `max_deapth` typo in the XGBoost parameters (should be `max_depth`). It is currently ignored.
- Sort rows by Driver, Race and `LapNumber` before computing `laps_since_pit`, since the feature depends on row order.
- Fit the target encoders on the training fold only, to avoid slight optimism in the validation score.
- Try additional features such as lagged lap-time deltas and tyre life relative to typical stint length for each compound.
- Add early stopping and hyperparameter tuning.
- Consider whether `scale_pos_weight` is needed when optimizing for AUC, since it distorts predicted probabilities.

## Acknowledgements

- Kaggle Playground Series for the dataset and competition.
- [XGBoost](https://xgboost.readthedocs.io/) and [scikit-learn](https://scikit-learn.org/).
