# EEG Multiclass Classification with PyTorch Lightning, Calibration, and XAI

This repository contains an end-to-end machine learning pipeline for multiclass
classification of EEG signal states. The project uses a neural network built
with PyTorch Lightning, evaluates calibrated probabilities, analyzes prediction
uncertainty, and explains model behavior with permutation feature importance
and LIME.

The complete workflow is implemented in
`EEG_project_presentation.ipynb`, with generated data splits, figures, metrics,
model checkpoints, and explanation artifacts stored in the repository.

## Task Description

The assignment requires the implementation of a full pipeline for building and
optimizing a neural-network-based multiclass classifier with PyTorch Lightning.
Using a dataset with at least 5,000 examples, the project should include:

1. Data cleaning, including handling missing values and outliers.
2. Exploratory Data Analysis.
3. Implementation, hyperparameter optimization, and training of a classifier
   with PyTorch Lightning.
4. Probability calibration of the trained classifier.
5. Evaluation of classification uncertainty, including the selection of the
   most uncertain and most confident cases.
6. Evaluation of explanatory-variable importance with permutation feature
   importance, computed separately for the training and test sets, followed by
   conclusions.
7. Global feature-importance analysis with LIME, comparison with permutation
   feature importance, and conclusions.
8. Local LIME explanations, including a comparison of feature weights for
   confident and uncertain cases, followed by conclusions.

## Problem Overview

Electroencephalography (EEG) records electrical activity produced by the brain.
EEG signals are temporal, noisy, and highly variable across patients and
recording conditions. In machine learning tasks, this means that the model must
learn discriminative signal patterns from sequences of amplitudes rather than
from conventional tabular clinical variables.

In this project, every observation is a short EEG segment represented by 178
consecutive signal amplitudes:

```text
EEG segment -> 178 time points -> one EEG state label
```

The goal is to predict one of five EEG states:

| Class | Label | Meaning |
|---:|---|---|
| 1 | `epileptic_seizure` | EEG activity during an epileptic seizure |
| 2 | `tumor_area_eeg` | EEG from the tumor area |
| 3 | `healthy_area_eeg` | EEG from a healthy brain area of a patient with a tumor |
| 4 | `eyes_closed` | EEG with eyes closed |
| 5 | `eyes_open` | EEG with eyes open |

Because the features `X1` to `X178` are ordered time points, model explanations
should be interpreted as important signal positions or signal segments, not as
independent clinical variables.

## Dataset

The project uses the public **Epileptic Seizure Recognition** dataset:

[https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition)

Dataset characteristics:

| Property | Value |
|---|---:|
| Samples | 11,500 |
| Signal features | 178 |
| Target classes | 5 |
| Samples per class | 2,300 |
| Missing values | 0 |
| Duplicates | 0 |

The original assignment points to OpenML as the preferred source, but the
available EEG dataset there is binary, while this task requires multiclass
classification. This project therefore uses a public five-class EEG dataset
that satisfies the multiclass requirement.

## Repository Structure

```text
.
|-- EEG_project_presentation.ipynb      # Main notebook with the complete pipeline
|-- README.md                           # Project documentation
|-- requirements.txt                    # Python dependencies
|-- data/
|   |-- README.md                       # Detailed data documentation
|   |-- raw/
|   |   `-- data.csv                    # Raw EEG dataset
|   `-- processed/
|       |-- train.csv                   # Training split after preprocessing
|       |-- val.csv                     # Validation split after preprocessing
|       `-- test.csv                    # Test split after preprocessing
|-- outputs/
|   |-- figures/                        # EDA, evaluation, calibration, and XAI plots
|   |-- lime/                           # Local LIME explanations as HTML files
|   |-- metrics/                        # Test metrics and classification report
|   |-- models/                         # Model checkpoints and calibration artifacts
|   `-- tables/                         # Intermediate and final analytical tables
```

## Methodology

The pipeline follows a complete supervised-learning workflow.

### 1. Data Cleaning and Preprocessing

The preprocessing stage:

- removes the technical identifier column `Unnamed`,
- verifies missing values and duplicates,
- remaps labels from `1-5` to `0-4` for `CrossEntropyLoss`,
- performs a stratified train/validation/test split with a `70/15/15` ratio,
- clips extreme values using the `0.001` and `0.999` quantiles computed only on
  the training set,
- standardizes features with `StandardScaler` fitted only on the training set.

The final processed files contain:

```text
X1, X2, ..., X178, target
```

### 2. Exploratory Data Analysis

The EDA stage investigates:

- class distribution,
- example EEG signals for each class,
- average signal shape per class,
- signal energy by class,
- PCA projection of the training data.

The analysis shows that classes are perfectly balanced. The epileptic seizure
class has the most distinctive amplitude and energy profile, while the tumor
area and healthy area EEG classes are more similar and therefore harder to
separate.

Key figures are available in `outputs/figures/`:

- `class_distribution.png`
- `example_signals.png`
- `mean_signals.png`
- `signal_energy_by_class.png`
- `pca_train.png`

### 3. Model

The classifier is a multilayer perceptron implemented with PyTorch Lightning.
The input layer receives 178 EEG time points and the output layer predicts one
of five classes.

Model structure:

```text
178 EEG time points
-> Linear + BatchNorm + ReLU + Dropout
-> Linear + BatchNorm + ReLU + Dropout
-> Linear + BatchNorm + ReLU + Dropout
-> Linear output layer with 5 classes
```

Training setup:

| Component | Value |
|---|---|
| Framework | PyTorch Lightning |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW |
| Scheduler | ReduceLROnPlateau |
| Hyperparameter search | Optuna |
| Early stopping monitor | validation macro F1 |

Best hyperparameters:

```json
{
  "dropout": 0.1,
  "learning_rate": 0.000679733642345161,
  "weight_decay": 0.00001176549814968483,
  "hidden_dims": [512, 256, 128]
}
```

### 4. Probability Calibration

Neural networks can produce overconfident probabilities. To improve probability
quality, the trained model is calibrated with temperature scaling fitted on the
validation set.

Calibration keeps the predicted class order unchanged, but adjusts the
probability distribution so that confidence better matches empirical accuracy.
The reliability diagram is stored in:

```text
outputs/figures/reliability_diagram.png
```

### 5. Uncertainty Analysis

For each test prediction, the project computes:

- maximum predicted probability,
- entropy of the class distribution,
- margin between the top-1 and top-2 probabilities.

These values are used to select:

- the most confident correct predictions,
- the most uncertain correct predictions,
- the most uncertain incorrect predictions.

The resulting tables are stored in `outputs/tables/` and are later used for
local LIME explanations.

### 6. Explainable AI

Two XAI methods are used.

**Permutation Feature Importance**

PFI measures how much macro F1 decreases when a feature is randomly permuted.
It is computed separately for the training and test sets. Because EEG time
points are correlated, the project also aggregates importance into temporal
segments:

```text
X1-X30, X31-X60, X61-X90, X91-X120, X121-X150, X151-X178
```

**LIME**

LIME is used in two ways:

- globally, by aggregating many local explanations,
- locally, by explaining selected confident and uncertain predictions.

Local HTML explanations are stored in:

```text
outputs/lime/
```

PFI and LIME are not expected to produce identical rankings, because they answer
different questions. Together, they show that the model uses multiple parts of
the EEG signal, with strong importance often assigned to the beginning of the
sequence and selected middle or later segments.

## Results

Final test metrics after temperature scaling:

| Metric | Value |
|---|---:|
| Accuracy | 0.758 |
| Balanced accuracy | 0.758 |
| Macro F1 | 0.758 |
| Weighted F1 | 0.758 |
| Macro OVR AUROC | 0.945 |
| NLL | 0.589 |
| Brier score | 0.338 |
| ECE | 0.032 |

Class-level F1 scores:

| Class | F1 score |
|---|---:|
| `epileptic_seizure` | 0.960 |
| `tumor_area_eeg` | 0.641 |
| `healthy_area_eeg` | 0.667 |
| `eyes_closed` | 0.783 |
| `eyes_open` | 0.741 |

The model performs best on the epileptic seizure class, which is consistent with
EDA findings showing that this class has the most distinctive signal profile.
The most difficult distinction is between tumor-area EEG and healthy-area EEG
from patients with tumors.

## Key Artifacts

| Path | Description |
|---|---|
| `EEG_project_presentation.ipynb` | Main executable project notebook |
| `data/README.md` | Detailed data documentation |
| `outputs/metrics/test_metrics.csv` | Final test metrics before and after calibration |
| `outputs/metrics/classification_report.csv` | Per-class precision, recall, and F1 |
| `outputs/models/best_hyperparameters.json` | Best Optuna configuration |
| `outputs/models/eeg_mlp_state_dict.pt` | Saved model weights |
| `outputs/models/temperature_scaler.joblib` | Fitted calibration artifact |
| `outputs/figures/confusion_matrix.png` | Confusion matrix |
| `outputs/figures/reliability_diagram.png` | Calibration plot |
| `outputs/tables/pfi_train.csv` | Training-set permutation feature importance |
| `outputs/tables/pfi_test.csv` | Test-set permutation feature importance |
| `outputs/tables/lime_global.csv` | Aggregated global LIME ranking |

## Reproducibility

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the main notebook:

```powershell
jupyter lab EEG_project_presentation.ipynb
```

The notebook is designed as the main execution entry point. It includes data
loading, preprocessing, EDA, model training, hyperparameter optimization,
calibration, uncertainty analysis, and XAI.

## Limitations

This project is educational and should not be treated as a clinical diagnostic
system. The public CSV dataset does not provide full patient/session metadata,
so a random split may mix samples that are not fully independent at the
patient-level. XAI results should be interpreted as signal-level evidence
within this dataset and model, not as direct neurophysiological conclusions.
