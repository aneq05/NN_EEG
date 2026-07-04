# Notatka do notebooka EEG

## Cel projektu

Notebook realizuje pełny pipeline klasyfikacji wieloklasowej sygnałów EEG:

1. wczytanie i przygotowanie danych,
2. eksploracyjna analiza danych,
3. budowa i trening modelu MLP w PyTorch Lightning,
4. strojenie hiperparametrów w Optunie,
5. kalibracja prawdopodobieństw,
6. analiza niepewności predykcji,
7. interpretacja modelu przez PFI i LIME.

Model ma rozpoznawać 5 klas sygnałów EEG na podstawie 178 punktów czasowych sygnału.

## Co dzieje się krok po kroku

### 1. Importy i stałe

Na początku notebook importuje biblioteki do:

- pracy z danymi: `pandas`, `numpy`,
- wykresów: `matplotlib`, `seaborn`,
- uczenia maszynowego: `scikit-learn`,
- deep learningu: `torch`, `lightning`,
- strojenia hiperparametrów: `optuna`,
- wyjaśnialności: `lime`.

Potem ustawiane są:

- ścieżki do folderów `data/`, `outputs/`, `reports/`,
- `SEED`, czyli ziarno losowości,
- słownik `LABELS`, który mapuje numery klas na ich znaczenie.

To daje powtarzalność eksperymentu i porządek w zapisie wyników.

### 2. Wczytanie danych

Dane są wczytywane z pliku CSV:

```python
raw = pd.read_csv(RAW_PATH)
```

Każdy wiersz to jedna próbka EEG:

- kolumny `X1-X178` są kolejnymi punktami czasowymi sygnału,
- kolumna `y` to etykieta klasy,
- kolumna `Unnamed` to identyfikator techniczny, nie cecha modelu.

### 3. Cleaning i preprocessing

Najważniejsze funkcje:

`clean_dataframe(df)`

- usuwa kolumny techniczne,
- usuwa duplikaty,
- tworzy `target = y - 1`, żeby klasy miały indeksy `0-4`.

`feature_columns(df)`

- wybiera tylko kolumny `X1-X178`.

`prepare_model_inputs(df)`

- rozdziela dane na `X` i `y`,
- ustawia typy zgodne z PyTorch:
  `float32` dla cech i `int64` dla targetu.

`split_train_val_test(x, y, ...)`

- robi stratyfikowany podział na:
  `train`, `validation`, `test`,
- dba o zachowanie proporcji klas.

`clip_outliers_from_train(...)`

- liczy granice przycięcia tylko na zbiorze treningowym,
- przycina wartości skrajne w train, val i test,
- dzięki temu nie ma leakage z walidacji i testu.

`scale_features(...)`

- dopasowuje `StandardScaler` tylko na `train`,
- przekształca `train`, `val`, `test`.

`build_splits(...)`

- składa wszystko do jednego słownika `splits`.

`save_processed_splits(...)`

- zapisuje `train.csv`, `val.csv`, `test.csv` po preprocessingu.

`split_and_scale(df, ...)`

- jest funkcją-orchestratorem,
- uruchamia wszystkie kroki po kolei i zwraca gotowy `splits`.

Znaczenie `splits`:

```python
{
    "x_train": ...,
    "y_train": ...,
    "x_val": ...,
    "y_val": ...,
    "x_test": ...,
    "y_test": ...,
    "feature_names": ...
}
```

To jest główny obiekt z danymi, z którego później korzysta model, kalibracja i XAI.

### 4. EDA

Funkcja:

`run_eda_inline(df, splits)`

Tworzy kilka wykresów:

- rozkład liczebności klas,
- pojedynczy przykładowy sygnał z każdej klasy,
- średni sygnał dla każdej klasy,
- boxplot energii sygnału według klasy,
- PCA 2D dla danych treningowych po skalowaniu.

Dodatkowo tworzy proste statystyki sygnału:

- średnia,
- odchylenie standardowe,
- minimum,
- maksimum,
- energia sygnału.

Cel EDA:

- sprawdzić balans klas,
- zobaczyć, czy sygnały różnią się kształtem i amplitudą,
- sprawdzić, czy klasy są choć trochę separowalne.

## Kod modelu

### 5. `EEGDataModule`

Klasa:

`EEGDataModule(L.LightningDataModule)`

Po co jest potrzebna:

- opakowuje gotowe tablice NumPy do `TensorDataset`,
- tworzy `DataLoader` dla train, val i test,
- pozwala Lightningowi wygodnie obsługiwać dane.

Najważniejsze elementy:

`__init__(splits, batch_size)`

- przyjmuje gotowy słownik `splits`,
- zapamiętuje wielkość batcha.

`setup()`

- buduje trzy datasety:
  `train_ds`, `val_ds`, `test_ds`.

`_dataset(x, y)`

- zamienia NumPy na tensory PyTorch.

`train_dataloader()`, `val_dataloader()`, `test_dataloader()`

- zwracają `DataLoader`,
- `shuffle=True` jest tylko dla treningu,
- `num_workers=0` zostało ustawione dla prostoty i stabilności notebooka.

### 6. Konfiguracja modelu

`ModelConfig`

- przechowuje podstawowe hiperparametry modelu:
  liczbę wejść, liczbę klas, architekturę, dropout, learning rate i weight decay.

### 7. Model `EEGMLP`

Klasa:

`EEGMLP(L.LightningModule)`

To główny model sieci neuronowej.

#### `__init__(cfg)`

- buduje architekturę MLP,
- każda warstwa ukryta składa się z:
  `Linear -> BatchNorm1d -> ReLU -> Dropout`,
- na końcu dodawana jest warstwa wyjściowa z 5 neuronami.

`hidden_dims`

- określa liczbę neuronów w kolejnych warstwach ukrytych,
- np. `[512, 256, 128]` oznacza 3 warstwy ukryte.

#### `forward(x)`

- przepuszcza dane przez `self.net`,
- zwraca logity.

#### `training_step(batch, batch_idx)`

- liczy logity,
- liczy `CrossEntropyLoss`,
- liczy accuracy dla batcha,
- loguje `train_loss` i `train_acc`,
- zwraca stratę do optymalizacji.

#### `validation_step(batch, batch_idx)`

- liczy stratę walidacyjną,
- liczy accuracy,
- liczy `macro_f1`,
- loguje `val_loss`, `val_acc`, `val_macro_f1`.

To `val_macro_f1` jest potem monitorowane przez Optunę, `EarlyStopping`, scheduler i checkpoint.

#### `configure_optimizers()`

- ustawia optymalizator `AdamW`,
- ustawia scheduler `ReduceLROnPlateau`,
- scheduler zmniejsza learning rate, gdy `val_macro_f1` przestaje się poprawiać.

Znaczenie:

- `optimizer` odpowiada za aktualizację wag,
- `scheduler` odpowiada za sterowanie tempem uczenia.

### 8. Funkcje pomocnicze modelu

`make_model(overrides=None)`

- tworzy obiekt `EEGMLP` z domyślnych albo nadpisanych hiperparametrów.

`trainer_for(callbacks=None, checkpointing=False, quiet=False)`

- tworzy obiekt `L.Trainer`,
- to lightningowy menedżer treningu,
- kontroluje liczbę epok, callbacki, pasek postępu, checkpointy i deterministyczność.

## Predykcja, kalibracja i ewaluacja

### 9. Predykcja

`softmax(logits)`

- zamienia logity na prawdopodobieństwa klas.

`predict_logits(model, x, batch_size=512)`

- uruchamia model w trybie ewaluacji,
- liczy logity partiami,
- zwraca wynik jako `numpy.ndarray`.

### 10. Temperature scaling

Klasa:

`TemperatureScaler`

Po co jest potrzebna:

- model po treningu bywa zbyt pewny swoich predykcji,
- temperature scaling kalibruje prawdopodobieństwa bez zmiany samej kolejności klas.

#### `fit(logits, y_true)`

- szuka najlepszej temperatury `T`,
- robi to na zbiorze walidacyjnym,
- optymalizuje `log_loss`.

Jeśli `T > 1`, rozkład prawdopodobieństw staje się bardziej płaski, a model mniej pewny.

#### `transform_logits(logits)`

- dzieli logity przez wyuczoną temperaturę.

### 11. Metryki

`expected_calibration_error(...)`

- mierzy różnicę między pewnością modelu a faktyczną trafnością.

`multiclass_brier(...)`

- liczy błąd prawdopodobieństw w sensie Brier score.

`metrics_dict(...)`

- zwraca najważniejsze metryki:
  accuracy, balanced accuracy, macro F1, weighted F1, NLL, Brier, ECE, macro OVR AUROC.

`save_reliability_diagram(...)`

- tworzy wykres kalibracji przed i po scalingu.

`save_evaluation(...)`

- zapisuje metryki do CSV,
- zapisuje classification report,
- tworzy confusion matrix,
- tworzy reliability diagram.

## Trening i strojenie hiperparametrów

### 12. Ustawienia eksperymentu

W notebooku są stałe:

- `FORCE_OPTUNA_SEARCH`,
- `N_OPTUNA_TRIALS`,
- `MAX_EPOCHS`,
- `BATCH_SIZE`,
- `RUN_LIME`.

Pozwalają sterować, czy notebook:

- ma odtwarzać tuning od zera,
- ile prób ma zrobić Optuna,
- ile epok ma trwać trening,
- czy ma uruchamiać LIME.

### 13. Przygotowanie treningu

Tworzony jest:

```python
data_module = EEGDataModule(splits, batch_size=BATCH_SIZE)
```

oraz ścieżki do zapisów:

- `best_hyperparameters.json`,
- `hyperparameter_search.csv`.

### 14. Strojenie w Optunie

Jeśli istnieją zapisane najlepsze parametry i nie wymuszamy nowego strojenia, notebook je wczyta.

W przeciwnym razie:

- definiuje funkcję `objective(trial)`,
- wybiera architekturę `small`, `medium` lub `large`,
- losuje `dropout`, `learning_rate`, `weight_decay`,
- trenuje model,
- zwraca `val_macro_f1`.

Na tej podstawie Optuna:

- porównuje próby,
- zapisuje wyniki do tabeli,
- zapisuje najlepszy zestaw hiperparametrów.

### 15. Finalny trening

Po wybraniu `best_params` model trenowany jest jeszcze raz:

- z `ModelCheckpoint`,
- z `EarlyStopping`,
- na tych najlepszych ustawieniach.

Jeśli jest zapisany najlepszy checkpoint, notebook go wczytuje i używa go jako finalnego modelu.

Na końcu zapisuje:

```python
torch.save(model.state_dict(), ...)
```

czyli same wagi modelu.

### 16. Kalibracja i ewaluacja

Po treningu notebook:

1. liczy logity dla walidacji i testu,
2. dopasowuje `TemperatureScaler` na walidacji,
3. liczy prawdopodobieństwa przed i po kalibracji,
4. zapisuje metryki i raport klasyfikacji,
5. zapisuje pełne prawdopodobieństwa predykcji testowych.

To jest główny moment, w którym powstają finalne wyniki modelu.

## Analiza niepewności

### 17. Funkcje do niepewności

`uncertainty_table(y_true, probs)`

- dla każdej próbki zapisuje:
  `max_probability`, `entropy`, `margin_top1_top2`,
- pozwala ocenić, czy model był pewny czy niepewny.

`save_uncertainty_cases(table, n=20)`

- wybiera:
  najbardziej pewne poprawne przypadki,
  najbardziej niepewne poprawne przypadki,
  najbardziej niepewne błędne przypadki,
- zapisuje je do CSV.

Po co to jest potrzebne:

- pozwala zidentyfikować, gdzie model jest stabilny,
- przygotowuje przypadki do lokalnej interpretacji LIME.

## XAI: PFI i LIME

### 18. PFI

`permutation_importance_inline(...)`

- miesza jedną cechę na raz,
- sprawdza, jak bardzo spada `macro_f1`,
- im większy spadek, tym cecha ważniejsza.

`segment_importance(...)`

- łączy pojedyncze punkty czasowe w segmenty, np. `X1-X30`,
- daje bardziej stabilną interpretację niż patrzenie na pojedyncze `X`.

`save_importance_plot(...)`

- rysuje wykres ważności cech lub segmentów.

### 19. LIME

`build_lime_explainer(...)`

- buduje obiekt LIME na podstawie danych treningowych.

`map_lime_feature(...)`

- mapuje tekstowy opis cechy z LIME z powrotem na nazwę `X...`.

`global_lime_inline(...)`

- liczy wiele lokalnych wyjaśnień,
- agreguje bezwzględne wagi,
- tworzy ranking globalny.

`save_lime_plot(df)`

- rysuje wykres globalnej ważności LIME.

`save_local_lime_inline(...)`

- zapisuje lokalne wyjaśnienia jako HTML,
- robi to dla przypadków pewnych i niepewnych.

### 20. Uruchomienie XAI

W tej części notebook:

- definiuje `predict_fn` i `predict_proba_fn`,
- liczy PFI dla train i test,
- liczy PFI segmentowe,
- liczy LIME globalne,
- zapisuje wyjaśnienia lokalne.

To dzieje się dopiero po treningu, kalibracji i analizie niepewności, bo XAI potrzebuje już gotowego modelu i gotowych przypadków testowych.

## Najważniejsze zależności między częściami notebooka

- `clean_dataframe()` przygotowuje dane wejściowe do `split_and_scale()`.
- `split_and_scale()` tworzy `splits`, czyli centralny obiekt z danymi.
- `splits` trafia do `EEGDataModule`.
- `EEGDataModule` zasila `Trainer`.
- `EEGMLP` definiuje model, a `Trainer` uruchamia jego trening.
- Optuna szuka najlepszych hiperparametrów dla `EEGMLP`.
- `TemperatureScaler` kalibruje już wytrenowany model.
- `uncertainty_table()` przygotowuje przypadki dla LIME lokalnego.
- PFI i LIME interpretują finalny, skalibrowany model.

## Po co ten podział ma sens

Każda część notebooka odpowiada za inny etap pipeline'u:

- preprocessing przygotowuje poprawne dane,
- EDA pomaga zrozumieć problem,
- model i trainer uczą klasyfikator,
- kalibracja poprawia wiarygodność prawdopodobieństw,
- analiza niepewności pokazuje, gdzie model się waha,
- XAI pokazuje, z jakich fragmentów sygnału model korzysta.

To daje pełny i logiczny przepływ od danych surowych do interpretowalnych wyników końcowych.
