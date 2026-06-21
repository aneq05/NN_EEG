# Klasyfikacja stanow EEG z wykorzystaniem sieci neuronowej, kalibracji i XAI

## 1. Cel projektu

Celem projektu jest zbudowanie klasyfikatora wieloklasowego, ktory na podstawie
178 kolejnych punktow sygnalu EEG przewiduje jedna z pieciu klas stanu EEG.
Projekt obejmuje cleaning, EDA, trening sieci neuronowej w PyTorch Lightning,
optymalizacje hiperparametrow, kalibracje prawdopodobienstw, analize
niepewnosci oraz wyjasnialnosc z wykorzystaniem PFI i LIME.

## 2. Dane i przygotowanie

Wykorzystano zbior `Epileptic Seizure Recognition`:

- liczba probek: 11 500,
- liczba cech: 178,
- liczba klas: 5,
- rozklad klas: po 2300 probek w kazdej klasie,
- braki danych: 0,
- duplikaty: 0.

Usunieto kolumne identyfikacyjna `Unnamed`. Target `y` zostal przekodowany z
zakresu 1-5 na zakres 0-4. Dane podzielono stratyfikowanie na train/validation/test
w proporcji 70/15/15. Standaryzator dopasowano tylko na zbiorze treningowym.
Dodatkowo ekstremalne wartosci zostaly przyciete wedlug kwantyli 0.001 i 0.999
wyznaczonych na train, aby ograniczyc wplyw outlierow bez usuwania probek.

## 3. EDA

Wygenerowano:

- rozklad klas: `outputs/figures/class_distribution.png`,
- przykladowe sygnaly dla klas: `outputs/figures/example_signals.png`,
- srednie sygnaly klas: `outputs/figures/mean_signals.png`,
- energie sygnalu wedlug klasy: `outputs/figures/signal_energy_by_class.png`,
- wizualizacje PCA: `outputs/figures/pca_train.png`.

EDA pokazuje, ze klasy sa idealnie zbalansowane, a sygnaly klasy napadowej
wyrozniaja sie amplituda i ksztaltem. Klasy 2 i 3 sa bardziej podobne, co
pozniej widac w macierzy pomylek.

## 4. Model i trening

Model glowny to MLP w PyTorch Lightning:

```text
178 wejsc
Linear + BatchNorm + ReLU + Dropout
Linear + BatchNorm + ReLU + Dropout
Linear + BatchNorm + ReLU + Dropout
Linear -> 5 klas
```

Optymalizacja hiperparametrow zostala wykonana przez Optuna. Najlepsza
konfiguracja:

```json
{
  "dropout": 0.2,
  "learning_rate": 0.0001749661369974354,
  "weight_decay": 0.00024445012556986674,
  "hidden_dims": [512, 256, 128]
}
```

## 5. Wyniki

Metryki testowe po temperature scaling:

| Metryka | Wartosc |
|---|---:|
| accuracy | 0.720 |
| balanced accuracy | 0.720 |
| macro F1 | 0.718 |
| weighted F1 | 0.718 |
| macro OVR AUROC | 0.939 |
| NLL | 0.601 |
| Brier score | 0.356 |
| ECE | 0.031 |

Najlepsze wyniki osiagnieto dla klasy `epileptic_seizure`: F1 = 0.954.
Najslabsze wyniki wystapily dla klasy `tumor_area_eeg`: F1 = 0.509. Model
czesto myli klasy 2 i 3, co jest spodziewane, poniewaz obie dotycza zapisow
zwiazanych z pacjentami z guzem, ale roznych obszarow.

## 6. Kalibracja prawdopodobienstw

Zastosowano temperature scaling dopasowany na zbiorze walidacyjnym. Wyznaczona
temperatura wyniosla ok. 1.052. Kalibracja lekko poprawila NLL i Brier score:

| Etap | NLL | Brier | ECE |
|---|---:|---:|---:|
| przed kalibracja | 0.6018 | 0.3568 | 0.0253 |
| po kalibracji | 0.6015 | 0.3564 | 0.0308 |

Klasy decyzyjne nie zmienily sie, ale prawdopodobienstwa zostaly delikatnie
wygladzone. Diagram wiarygodnosci zapisano w
`outputs/figures/reliability_diagram.png`.

## 7. Niepewnosc

Dla kazdej probki testowej policzono:

- maksymalne prawdopodobienstwo,
- entropie rozkladu klas,
- margines miedzy top-1 i top-2.

Zapisano przypadki:

- `outputs/tables/most_confident_correct.csv`,
- `outputs/tables/most_uncertain_correct.csv`,
- `outputs/tables/most_uncertain_wrong.csv`.

Te grupy zostaly potem wykorzystane do lokalnych wyjasnien LIME.

## 8. Permutation Feature Importance

PFI policzono osobno dla train i test, zgodnie z wymaganiem. Najwazniejsze cechy
testowe to m.in. `X16`, `X24`, `X106`, `X68`, `X11`, `X10`, `X7`, `X12` i `X87`.

PFI segmentowe dla test wskazuje nastepujaca kolejnosc fragmentow sygnalu:

| Segment | Suma waznosci |
|---|---:|
| X1-X30 | 0.659 |
| X61-X90 | 0.558 |
| X91-X120 | 0.533 |
| X31-X60 | 0.527 |
| X121-X150 | 0.523 |
| X151-X178 | 0.493 |

Wniosek: model korzysta z informacji rozlozonej po calym sygnale, ale poczatkowy
fragment X1-X30 jest szczegolnie istotny w zbiorze testowym.

## 9. LIME

LIME uruchomiono globalnie przez agregacje lokalnych wyjasnien oraz lokalnie dla
przypadkow pewnych i niepewnych. Globalnie najwyzej pojawily sie cechy z
poczatku sygnalu, m.in. `X7`, `X4`, `X1`, `X9`, `X8`, `X2`, `X3`.

Porownanie z PFI jest spojne: obie metody wskazuja na duze znaczenie wczesnych
punktow czasowych. PFI dodatkowo pokazuje istotnosc pozniejszych punktow, np.
`X106` i segmentow srodkowych.

Lokalne wyjasnienia HTML zapisano w `outputs/lime/`. Dla przypadkow pewnych
wagi LIME zwykle koncentruja sie na kilku cechach wspierajacych jedna klase.
Dla przypadkow niepewnych widac slabszy margines decyzji i cechy wspierajace
konkurencyjne klasy.

## 10. Ograniczenia

Projekt ma charakter edukacyjny i nie jest systemem diagnostycznym. Random split
moze mieszac probki pochodzace z podobnych zrodel, poniewaz publiczny CSV nie
udostepnia pelnej informacji o pacjencie/sesji. Interpretacja cech `X1`-`X178`
dotyczy punktow czasowych sygnalu, a nie bezposrednio struktur biologicznych.

## 11. Reprodukowalnosc

Glowny pipeline:

```powershell
python -m src.run_pipeline --config configs/config.yaml
```

Najwazniejsze wyniki sa w `outputs/`, a konfiguracja w `configs/config.yaml`.

