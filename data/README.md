# Opis danych

Ten folder zawiera dane wykorzystane w projekcie klasyfikacji wieloklasowej
stanow EEG.

## 1. Zrodlo danych

Uzyty zbior danych to:

```text
Epileptic Seizure Recognition
```

Zbior jest dostepny publicznie na Kaggle:

```text
https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition
```

W projekcie dane sa pobierane automatycznie przez `kagglehub`, jezeli plik
`data/raw/data.csv` nie istnieje.

Uwaga: polecenie laboratoryjne wskazuje OpenML, ale dostepny na OpenML zbior EEG
jest binarny, a zadanie wymaga klasyfikacji wieloklasowej. Dlatego wybrano ten
publiczny, piecioklasowy zbior EEG.

## 2. Dane surowe

Surowe dane znajduja sie w:

```text
data/raw/data.csv
```

Plik surowy ma:

```text
11500 wierszy
180 kolumn
```

Kolumny:

| Kolumna | Znaczenie |
|---|---|
| `Unnamed` | identyfikator probki, nie jest cecha modelowa |
| `X1`-`X178` | kolejne punkty czasowe sygnalu EEG |
| `y` | oryginalna klasa probki |

Najwazniejsze jest to, ze kolumny `X1`, `X2`, ..., `X178` nie sa opisowymi
cechami klinicznymi. Sa to kolejne wartosci amplitudy sygnalu EEG w krotkim
fragmencie czasowym.

Jeden wiersz mozna interpretowac tak:

```text
jeden fragment EEG -> 178 kolejnych wartosci sygnalu -> jedna klasa stanu EEG
```

Czyli model nie dostaje np. wieku pacjenta, BMI albo nazwy pasma fal mozgowych.
Model dostaje surowy ksztalt sygnalu:

```text
[X1, X2, X3, ..., X178]
```

## 3. Klasy

Zmienna docelowa w danych surowych to:

```text
y
```

Ma piec klas:

| Klasa `y` | Znaczenie |
|---:|---|
| 1 | aktywnosc napadowa, epileptic seizure |
| 2 | zapis EEG z obszaru guza |
| 3 | zapis EEG ze zdrowego obszaru mozgu pacjenta z guzem |
| 4 | oczy zamkniete |
| 5 | oczy otwarte |

Rozklad klas w surowym zbiorze:

| Klasa | Liczba probek |
|---:|---:|
| 1 | 2300 |
| 2 | 2300 |
| 3 | 2300 |
| 4 | 2300 |
| 5 | 2300 |

Klasy sa idealnie zbalansowane. To jest korzystne, bo model nie jest od poczatku
faworyzowany w strone jednej dominujacej klasy.

## 4. Co jest brane do klasyfikacji

Do klasyfikacji brane sa tylko kolumny:

```text
X1, X2, ..., X178
```

Nie jest brana kolumna:

```text
Unnamed
```

bo jest tylko identyfikatorem probki.

Targetem jest kolumna:

```text
y
```

ale przed treningiem jest przekodowana do postaci wymaganej przez PyTorch:

| Oryginalne `y` | Target modelowy |
|---:|---:|
| 1 | 0 |
| 2 | 1 |
| 3 | 2 |
| 4 | 3 |
| 5 | 4 |

Powod:

```text
CrossEntropyLoss w PyTorch oczekuje etykiet klas od 0 do liczba_klas - 1.
```

## 5. Data cleaning

Cleaning wykonywany jest w pliku:

```text
src/preprocessing.py
```

Wykonane kroki:

1. Wczytanie `data/raw/data.csv`.
2. Usuniecie kolumny `Unnamed`.
3. Usuniecie ewentualnych kolumn zaczynajacych sie od `Unnamed`.
4. Sprawdzenie brakow danych.
5. Sprawdzenie duplikatow.
6. Utworzenie kolumny `target = y - 1`.

Wynik kontroli danych:

| Element | Wynik |
|---|---:|
| liczba wierszy | 11500 |
| liczba cech modelowych | 178 |
| brakujace wartosci | 0 |
| duplikaty | 0 |

Nie trzeba bylo usuwac wierszy z brakami danych, bo ich nie bylo.

## 6. Obsluga wartosci odstajacych

Poniewaz dane EEG moga miec duze amplitudy, w pipeline zastosowano lagodne
przyciecie ekstremalnych wartosci.

Uzyte kwantyle:

```text
0.001 i 0.999
```

Granice przyciecia sa liczone tylko na zbiorze treningowym. Potem te same
granice sa stosowane do walidacji i testu.

To ogranicza wplyw skrajnych amplitud i jednoczesnie zapobiega data leakage.

Granice sa zapisane w:

```text
data/processed/outlier_clip_bounds.joblib
```

## 7. Podzial danych

Dane sa dzielone stratyfikowanie po klasie, czyli kazdy split zachowuje podobny
rozklad klas.

Proporcje:

| Split | Udzial |
|---|---:|
| train | 70% |
| validation | 15% |
| test | 15% |

Pliki po podziale:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

Zbior treningowy sluzy do uczenia modelu.

Zbior walidacyjny sluzy do:

- wyboru hiperparametrow,
- early stopping,
- dopasowania kalibracji temperature scaling.

Zbior testowy sluzy tylko do koncowej oceny modelu.

## 8. Standaryzacja

Cechy `X1`-`X178` sa standaryzowane przez:

```text
StandardScaler
```

Scaler jest dopasowany tylko na zbiorze treningowym:

```text
scaler.fit(X_train)
```

Nastepnie jest stosowany do:

```text
X_train
X_val
X_test
```

Zapisany scaler:

```text
data/processed/scaler.joblib
```

To jest wazne, poniewaz model neuronowy zwykle uczy sie stabilniej, gdy cechy
maja podobna skale.

## 9. Dane po przetworzeniu

Po preprocessingu w `data/processed/` znajduja sie:

| Plik | Znaczenie |
|---|---|
| `train.csv` | dane treningowe po czyszczeniu, clippingu i standaryzacji |
| `val.csv` | dane walidacyjne po czyszczeniu, clippingu i standaryzacji |
| `test.csv` | dane testowe po czyszczeniu, clippingu i standaryzacji |
| `scaler.joblib` | zapisany `StandardScaler` |
| `outlier_clip_bounds.joblib` | zapisane granice przyciecia outlierow |
| `feature_names.joblib` | lista cech `X1`-`X178` |

W plikach `train.csv`, `val.csv` i `test.csv` znajduja sie:

```text
X1, X2, ..., X178, target
```

Czyli po preprocessingu nie ma juz kolumny `Unnamed`, a target jest w wersji
modelowej `0`-`4`.

## 10. Dlaczego klasyfikacja wieloklasowa

Nie robimy klasyfikacji binarnej typu:

```text
seizure vs non-seizure
```

Zadanie laboratoryjne wymaga klasyfikacji wieloklasowej, dlatego zostawiono
wszystkie piec klas.

Model rozroznia:

```text
epileptic seizure
tumor area EEG
healthy area EEG
eyes closed
eyes open
```

Dzieki temu problem jest trudniejszy i bardziej zgodny z wymaganiami zadania.

## 11. Jak interpretowac cechy XAI

Poniewaz `X1`-`X178` sa punktami czasowymi sygnalu, interpretacja PFI i LIME
dotyczy fragmentow sygnalu, a nie cech klinicznych.

Poprawna interpretacja:

```text
Model uznal, ze dany punkt lub segment czasowy sygnalu EEG byl istotny dla decyzji.
```

Mniej poprawna interpretacja:

```text
X7 oznacza konkretna ceche medyczna.
```

Dlatego w projekcie oprocz waznosci pojedynczych punktow zastosowano tez
segmenty czasowe:

```text
X1-X30
X31-X60
X61-X90
X91-X120
X121-X150
X151-X178
```

To pozwala mowic, ktory fragment sygnalu EEG byl najwazniejszy dla klasyfikacji.

