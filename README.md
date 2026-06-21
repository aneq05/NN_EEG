# EEG multiclass classification with calibration and XAI

Projekt realizuje checkpoint 2 z laboratorium 14-15: pelny pipeline klasyfikacji
wieloklasowej z uzyciem sieci neuronowej w PyTorch Lightning, kalibracji
prawdopodobienstw, analizy niepewnosci oraz metod XAI: permutation feature
importance i LIME.

## Temat

Klasyfikacja wieloklasowa stanow EEG na podstawie sygnalow
elektroencefalograficznych z wykorzystaniem sieci neuronowej, temperature
scaling oraz metod wyjasnialnej sztucznej inteligencji.

## Dane

Uzyty zbior: `Epileptic Seizure Recognition`.

- 11 500 probek.
- 178 cech sygnalowych `X1`-`X178`.
- 5 zbalansowanych klas po 2300 probek.
- Target oryginalny: `y` w zakresie 1-5.
- Target modelowy: `target = y - 1`.

Klasy:

| y | Znaczenie |
|---|---|
| 1 | epileptic_seizure |
| 2 | tumor_area_eeg |
| 3 | healthy_area_eeg |
| 4 | eyes_closed |
| 5 | eyes_open |

Uwaga formalna: polecenie laboratoryjne wskazuje OpenML. Sprawdzono OpenML API
dla tagu `eeg` oraz nazw `epileptic` i `seizure`. Dostepny dataset EEG na
OpenML (`eeg-eye-state`) jest binarny, a zadanie wymaga klasyfikacji
wieloklasowej. Dlatego zachowano temat EEG i wybrano publiczny, wieloklasowy
zbior Kaggle-compatible pobierany przez `kagglehub`.

## Struktura

```text
checkp2/
  configs/config.yaml
  data/raw/data.csv
  data/processed/
  src/
  outputs/
    figures/
    tables/
    metrics/
    models/
    lime/
  reports/final_report.md
```

## Uruchomienie

Z katalogu `checkp2`:

```powershell
python -m pip install -r requirements.txt
python -m src.run_pipeline --config configs/config.yaml
```

W srodowisku Codex uzyty byl Python:

```powershell
C:\Users\ankap\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m src.run_pipeline --config configs/config.yaml
```

Pipeline wykonuje:

1. Pobranie danych przez `kagglehub`, jesli `data/raw/data.csv` nie istnieje.
2. Cleaning: usuniecie kolumny identyfikacyjnej, kontrola brakow i duplikatow.
3. Split stratified train/val/test 70/15/15.
4. Winsoryzacje ekstremow na podstawie train oraz standaryzacje `StandardScaler`.
5. EDA i zapis wykresow.
6. Optuna search dla MLP Lightning.
7. Finalny trening modelu.
8. Temperature scaling na walidacji.
9. Ocene testowa przed i po kalibracji.
10. Analize niepewnosci.
11. PFI na train i test.
12. LIME globalnie i lokalnie.

## Wyniki z ostatniego runu

Najlepsze hiperparametry:

```json
{
  "dropout": 0.2,
  "learning_rate": 0.0001749661369974354,
  "weight_decay": 0.00024445012556986674,
  "hidden_dims": [512, 256, 128]
}
```

Metryki testowe po kalibracji:

- accuracy: 0.720
- balanced accuracy: 0.720
- macro F1: 0.718
- weighted F1: 0.718
- macro OVR AUROC: 0.939
- NLL: 0.601
- Brier score: 0.356
- ECE: 0.031

Najlepiej rozpoznawana jest klasa `epileptic_seizure` z F1 ok. 0.954.
Najtrudniejsza para klas to `tumor_area_eeg` i `healthy_area_eeg`, co widac
w confusion matrix.

## Najwazniejsze artefakty

- `outputs/metrics/test_metrics.csv` - metryki przed i po kalibracji.
- `outputs/metrics/classification_report.csv` - raport per class.
- `outputs/figures/confusion_matrix.png` - macierz pomylek.
- `outputs/figures/reliability_diagram.png` - kalibracja.
- `outputs/tables/hyperparameter_search.csv` - Optuna.
- `outputs/tables/pfi_train.csv`, `outputs/tables/pfi_test.csv` - PFI.
- `outputs/tables/pfi_segments_train.csv`, `outputs/tables/pfi_segments_test.csv` - PFI segmentowe.
- `outputs/tables/lime_global.csv` - globalna agregacja LIME.
- `outputs/lime/*.html` - lokalne wyjasnienia LIME.
- `outputs/models/eeg_mlp_state_dict.pt` - wagi finalnego modelu.
- `outputs/models/temperature_scaler.joblib` - kalibrator.
- `reports/final_report.md` oraz `reports/final_report.pdf` - raport do oddania.
