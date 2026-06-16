## Opis problemu i źródła danych

Projekt dotyczy klasyfikacji wieloklasowej kategorii bezpieczeństwa metod antykoncepcji na podstawie profilu zdrowotnego i reprodukcyjnego kobiet. Celem nie jest bezpośrednie rekomendowanie konkretnej metody antykoncepcji, lecz zbudowanie modelu, który dla pary:

```text
profil pacjentki + rozważana metoda antykoncepcji
````

przewiduje jedną z kategorii bezpieczeństwa medycznego:

```text
MEC 1 / MEC 2 / MEC 3 / MEC 4
```

Model ma więc charakter analityczny i edukacyjny. Nie zastępuje konsultacji lekarskiej i nie powinien być traktowany jako narzędzie podejmowania decyzji klinicznych.

---

## Źródła informacji wykorzystane w projekcie

W projekcie wykorzystywane są dwa źródła informacji:

### 1. NSFG 2022–2023 Female Respondent File

Podstawowym zbiorem danych jest publiczny zbiór:

```text
National Survey of Family Growth 2022–2023 — Female Respondent File
```

Źródło: CDC / National Center for Health Statistics.

Zbiór zawiera dane ankietowe dotyczące m.in.:

* cech demograficznych respondentek,
* historii ciąż i porodów,
* zdrowia reprodukcyjnego,
* stosowania antykoncepcji,
* płodności,
* relacji i życia rodzinnego,
* wybranych aspektów zdrowia ogólnego,
* dostępu do opieki zdrowotnej.

W projekcie dane NSFG są wykorzystywane do zbudowania profilu zdrowotno-reprodukcyjnego kobiety, np.:

```text
wiek
BMI
palenie
historia ciąż
liczba porodów
status ciąży
karmienie piersią
samoocena zdrowia
korzystanie z opieki zdrowotnej
wybrane informacje reprodukcyjne
```

### 2. U.S. Medical Eligibility Criteria for Contraceptive Use 2024

Drugim źródłem są wytyczne:

```text
U.S. Medical Eligibility Criteria for Contraceptive Use, 2024
```

Źródło: CDC.

U.S. MEC określa, czy dana metoda antykoncepcji jest odpowiednia dla osoby z określonymi cechami zdrowotnymi lub stanami klinicznymi.

W projekcie kategorie U.S. MEC są traktowane jako klasy docelowe modelu.

---

## Definicja klas targetu

Zmienna docelowa modelu to:

```text
mec_category
```

Przyjmuje ona jedną z czterech klas:

| Klasa | Znaczenie                                                                         |
| ----- | --------------------------------------------------------------------------------- |
| MEC 1 | metoda bez ograniczeń                                                             |
| MEC 2 | metoda zwykle bezpieczna, korzyści przeważają nad ryzykiem                        |
| MEC 3 | metoda zwykle niezalecana, chyba że inne opcje są nieakceptowalne lub niedostępne |
| MEC 4 | metoda przeciwwskazana, nieakceptowalne ryzyko                                    |

Problem jest więc klasyfikacją wieloklasową z czterema klasami.

---

## Konstrukcja zbioru modelowego

Oryginalny plik NSFG ma strukturę:

```text
jeden wiersz = jedna respondentka
```

Na potrzeby projektu zbiór jest przekształcany do struktury:

```text
jeden wiersz = jedna respondentka + jedna rozważana metoda antykoncepcji
```

Oznacza to, że dla każdej respondentki generowanych jest kilka rekordów — po jednym dla każdej analizowanej metody antykoncepcji.

Przykład:

| ID respondentki | Profil zdrowotny                                   | Rozważana metoda                | Klasa docelowa |
| --------------- | -------------------------------------------------- | ------------------------------- | -------------- |
| 001             | wiek 24, nie pali, brak istotnych czynników ryzyka | implant                         | MEC 1          |
| 001             | wiek 24, nie pali, brak istotnych czynników ryzyka | combined hormonal contraception | MEC 1          |
| 002             | wiek 38, pali, nadciśnienie                        | combined hormonal contraception | MEC 3 / MEC 4  |
| 002             | wiek 38, pali, nadciśnienie                        | copper IUD                      | MEC 1          |

Dzięki temu model uczy się zależności pomiędzy:

```text
cechami pacjentki
+
typem metody antykoncepcji
```

a kategorią bezpieczeństwa medycznego danej metody.

---

## Analizowane grupy metod antykoncepcji

W projekcie metody antykoncepcji są pogrupowane do kilku kategorii, aby ograniczyć problem bardzo rzadkich metod i ułatwić interpretację wyników.

Planowane grupy metod:

| Grupa metody        | Przykłady                              |
| ------------------- | -------------------------------------- |
| combined_hormonal   | tabletka dwuskładnikowa, plaster, ring |
| progestin_only_pill | tabletka jednoskładnikowa              |
| dmpa_injection      | zastrzyk antykoncepcyjny               |
| implant             | implant antykoncepcyjny                |
| copper_iud          | wkładka miedziana                      |
| lng_iud             | wkładka hormonalna                     |
| barrier_method      | prezerwatywa, metody barierowe         |

Zmienna opisująca rozważaną metodę antykoncepcji jest jedną ze zmiennych wejściowych modelu.

---

## Etykietowanie danych

Kategorie `MEC 1`, `MEC 2`, `MEC 3` i `MEC 4` nie są bezpośrednią kolumną w zbiorze NSFG. Są tworzone w procesie preprocessingu na podstawie uproszczonego odwzorowania reguł U.S. MEC.

Proces etykietowania wygląda następująco:

1. Z danych NSFG wybierane są cechy opisujące profil zdrowotny i reprodukcyjny respondentki.
2. Dla każdej respondentki generowane są rekordy odpowiadające rozważanym metodom antykoncepcji.
3. Dla każdej pary `respondentka + metoda` przypisywana jest kategoria MEC.
4. Powstała zmienna `mec_category` jest używana jako target w klasyfikacji wieloklasowej.

Przykładowa logika etykietowania:

```text
młoda osoba, brak istotnych czynników ryzyka + implant
→ MEC 1

osoba powyżej 35 lat, palenie + metoda estrogenowa
→ MEC 3 lub MEC 4

nadciśnienie + combined hormonal contraception
→ wyższa kategoria ryzyka

brak istotnych przeciwwskazań + copper IUD
→ MEC 1
```

W projekcie wykorzystywane jest uproszczone odwzorowanie wybranych reguł U.S. MEC, ograniczone do cech dostępnych w zbiorze NSFG. Oznacza to, że model nie obejmuje wszystkich możliwych stanów klinicznych opisanych w pełnych wytycznych.

---

## Zmienne wejściowe modelu

Model otrzymuje dwa typy informacji:

### 1. Profil respondentki

Przykładowe grupy zmiennych:

| Grupa            | Przykłady                                         |
| ---------------- | ------------------------------------------------- |
| demograficzne    | wiek, wykształcenie, status związku               |
| zdrowotne        | BMI, palenie, samoocena zdrowia                   |
| reprodukcyjne    | liczba ciąż, liczba porodów, aktualna ciąża       |
| opieka zdrowotna | ubezpieczenie, korzystanie z usług medycznych     |
| okołoporodowe    | karmienie piersią, czas od porodu, jeśli dostępne |

### 2. Rozważana metoda antykoncepcji

Przykład:

```text
combined_hormonal
implant
copper_iud
lng_iud
dmpa_injection
barrier_method
```

Na podstawie tych informacji model przewiduje kategorię:

```text
MEC 1 / MEC 2 / MEC 3 / MEC 4
```

---

## Schemat działania modelu

```text
profil zdrowotny kobiety
+
rozważana metoda antykoncepcji
        ↓
preprocessing danych
        ↓
kodowanie zmiennych kategorycznych
        ↓
standaryzacja zmiennych numerycznych
        ↓
sieć neuronowa MLP w PyTorch Lightning
        ↓
prawdopodobieństwa klas MEC 1–4
        ↓
kalibracja prawdopodobieństw
        ↓
analiza niepewności i XAI
```

Przykładowe wyjście modelu:

```text
MEC 1: 0.05
MEC 2: 0.18
MEC 3: 0.52
MEC 4: 0.25
```

W tym przypadku model klasyfikuje daną parę `profil + metoda` jako `MEC 3`, ale z istotnym prawdopodobieństwem klasy `MEC 4`, co może oznaczać przypadek niepewny lub graniczny.

---

## Uzasadnienie klasyfikacji wieloklasowej

Zadanie jest klasyfikacją wieloklasową, ponieważ model nie przewiduje wartości binarnej typu „bezpieczna / niebezpieczna”, lecz wybiera jedną z czterech kategorii medycznej kwalifikacji:

```text
MEC 1
MEC 2
MEC 3
MEC 4
```

Takie podejście jest bardziej informacyjne niż klasyfikacja binarna, ponieważ pozwala rozróżnić:

* brak ograniczeń,
* niewielkie lub akceptowalne ryzyko,
* sytuację wymagającą ostrożności,
* przeciwwskazanie do stosowania danej metody.

---

## Wyjaśnialność modelu

Projekt obejmuje dwie metody wyjaśnialności:

### Permutation Feature Importance

PFI pozwala ocenić, które zmienne najbardziej wpływają na jakość klasyfikacji. Ważność cech jest liczona osobno dla zbioru treningowego i testowego, aby sprawdzić stabilność modelu i wykryć ewentualne przeuczenie.

Przykładowe cechy, które mogą mieć wysoką ważność:

```text
typ metody antykoncepcji
wiek
palenie
BMI
status ciąży
karmienie piersią
wybrane zmienne reprodukcyjne
```

### LIME

LIME pozwala wyjaśniać pojedyncze predykcje modelu. W projekcie LIME jest wykorzystywany na dwa sposoby:

1. Globalnie — poprzez agregację wielu lokalnych wyjaśnień.
2. Lokalnie — poprzez analizę przypadków najbardziej pewnych i najbardziej niepewnych.

Przykładowo, dla przypadku niepewnego LIME może pokazać, że część cech wspiera klasę `MEC 3`, a część klasę `MEC 4`.

---

## Kalibracja i niepewność

Ponieważ sieci neuronowe mogą generować nadmiernie pewne prawdopodobieństwa, w projekcie wykonywana jest kalibracja prawdopodobieństw metodą temperature scaling.

Po kalibracji analizowana jest niepewność predykcji z wykorzystaniem:

```text
maksymalnego prawdopodobieństwa
entropii
marginesu między dwiema najbardziej prawdopodobnymi klasami
```

Na tej podstawie wybierane są przypadki:

```text
najbardziej pewne
najbardziej niepewne
błędne i niepewne
```

Następnie przypadki te są analizowane lokalnie z wykorzystaniem LIME.

---

## Ograniczenia projektu

Projekt ma charakter edukacyjny, predykcyjny i interpretacyjny.

Najważniejsze ograniczenia:

* model nie jest narzędziem medycznym,
* model nie zastępuje konsultacji z lekarzem,
* model nie powinien być używany do samodzielnego wyboru metody antykoncepcji,
* etykiety MEC są tworzone na podstawie uproszczonego odwzorowania wybranych reguł,
* uwzględniane są tylko cechy dostępne w zbiorze NSFG,
* dane NSFG mają charakter ankietowy i mogą zawierać błędy deklaratywne,
* model nie dowodzi związków przyczynowo-skutkowych.

Celem projektu jest pokazanie pełnego pipeline'u klasyfikacji wieloklasowej z wykorzystaniem sieci neuronowej, kalibracji prawdopodobieństw oraz metod XAI.
