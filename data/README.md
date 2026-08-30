# Data Documentation

This directory contains the data used by the EEG multiclass classification
workflow.

## Source

The project uses the public Kaggle dataset:

```text
Epileptic Seizure Recognition
https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition
```

The pipeline expects the raw CSV at:

```text
data/raw/data.csv
```

If the file is missing, `src.data.ensure_raw_data` can download it with
`kagglehub`. The dependency is included in `requirements.txt`.

## Dataset Choice

The dataset contains five balanced EEG classes, which makes it suitable for a
multiclass classification project rather than a binary seizure vs. non-seizure
setup.

| Property | Value |
|---|---:|
| Samples | 11,500 |
| Signal features | 178 |
| Target classes | 5 |
| Samples per class | 2,300 |
| Missing values | 0 |
| Duplicates | 0 |

## Raw Schema

The raw file has 180 columns:

| Column | Meaning |
|---|---|
| `Unnamed` | Technical sample identifier, removed before modeling |
| `X1`-`X178` | Consecutive EEG signal amplitudes |
| `y` | Original class label |

Each row represents one short EEG segment:

```text
one EEG segment -> 178 ordered signal values -> one EEG state label
```

The `X1`-`X178` columns are ordered time points, not independent clinical
variables.

Before preprocessing, `src.data.validate_raw_schema` verifies that:

- feature columns are exactly `X1` through `X178` in order,
- all EEG feature columns are numeric,
- `X1`-`X178` and `y` contain no missing values,
- the target column `y` contains exactly labels `1, 2, 3, 4, 5`.

## Classes

| Raw `y` | Model target | Meaning |
|---:|---:|---|
| 1 | 0 | Epileptic seizure activity |
| 2 | 1 | EEG from the tumor area |
| 3 | 2 | EEG from a healthy brain area of a patient with a tumor |
| 4 | 3 | Eyes closed |
| 5 | 4 | Eyes open |

PyTorch `CrossEntropyLoss` expects labels from `0` to `num_classes - 1`, so the
pipeline maps the raw `1-5` labels to `0-4`.

## Preprocessing

Preprocessing is implemented in `src/data.py` and run by `python -m src.train`.

Steps:

1. Load `data/raw/data.csv`, downloading it with `kagglehub` if needed.
2. Validate the raw schema.
3. Remove technical `Unnamed*` identifier columns.
4. Remove duplicate rows.
5. Create `target = y - 1`.
6. Create stratified train, validation, and test splits with a `70/15/15` ratio.
7. Compute outlier clipping bounds from the training set only.
8. Apply the same clipping bounds to train, validation, and test sets.
9. Fit `StandardScaler` on the training set only.
10. Transform train, validation, and test features with the fitted scaler.

Computing clipping bounds and scaling parameters only on the training split
prevents train-test leakage.

## Processed Files

After running the pipeline, `data/processed/` contains:

| File | Meaning |
|---|---|
| `train.csv` | Training split after clipping and scaling |
| `val.csv` | Validation split after clipping and scaling |
| `test.csv` | Test split after clipping and scaling |
| `scaler.joblib` | Fitted training-set `StandardScaler` |
| `outlier_clip_bounds.joblib` | Training-set clipping bounds |
| `feature_names.joblib` | Ordered list of `X1`-`X178` features |

The processed CSV files contain:

```text
X1, X2, ..., X178, target
```

`data/raw/` and `data/processed/` are intentionally ignored by Git because the
pipeline can regenerate them locally.

## XAI Interpretation

Permutation feature importance and LIME should be interpreted at the signal
level. A high-importance `X` feature means that the corresponding time point or
signal segment influenced the classifier for this dataset and model. It should
not be interpreted as a named medical variable.

The project also aggregates point-level importance into temporal segments:

```text
X1-X30
X31-X60
X61-X90
X91-X120
X121-X150
X151-X178
```
