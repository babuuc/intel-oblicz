import os
import pandas as pd

folder = os.path.dirname(os.path.abspath(__file__))
plik1 = folder + "/iris_big_no_errors.csv"
plik2 = folder + "/iris_big_normalized.csv"
plik3 = folder + "/iris_train.csv"
plik4 = folder + "/iris_test.csv"

if (os.path.exists(plik1)) == False:
    raise Exception("brak iris_big_no_errors.csv najpierw uruchom zad1.py")

dane = pd.read_csv(plik1)

numeryczne = []
numeryczne.append(dane.columns[0])
numeryczne.append(dane.columns[1])
numeryczne.append(dane.columns[2])
numeryczne.append(dane.columns[3])

cel = dane.columns[4]

# normalizacja min max do przedzialu 0 i 1
for i in range(len(numeryczne)):
    nazwa = numeryczne[i]
    min_wartosc = dane[nazwa].min()
    max_wartosc = dane[nazwa].max()

    if max_wartosc == min_wartosc:
        for j in dane.index:
            dane.loc[j, nazwa] = 0.0
    else:
        for j in dane.index:
            dane.loc[j, nazwa] = (dane[nazwa][j] - min_wartosc) / (max_wartosc - min_wartosc)

dane.to_csv(plik2, index=False)

# podzial stratified 80 20 na train test zachowanie proporcji klas
tablica1 = []
tablica2 = []

klasy = list(dane[cel].unique())
klasy.sort()

for i in range(len(klasy)):
    nazwa_klasy = klasy[i]
    grupa = dane[dane[cel] == nazwa_klasy]
    grupa = grupa.sample(frac=1.0, random_state=42)
    
    ciecie = int(0.8 * len(grupa))
    
    tablica1.append(grupa.iloc[:ciecie])
    tablica2.append(grupa.iloc[ciecie:])

dane_train = pd.concat(tablica1)
dane_train = dane_train.sample(frac=1.0, random_state=42)
dane_train = dane_train.reset_index(drop=True)

dane_test = pd.concat(tablica2)
dane_test = dane_test.sample(frac=1.0, random_state=42)
dane_test = dane_test.reset_index(drop=True)

dane_train.to_csv(plik3, index=False)
dane_test.to_csv(plik4, index=False)

print("zapisano " + plik2.split("/")[-1])
print("zapisano " + plik3.split("/")[-1] + " (" + str(len(dane_train)) + " rekordow)")
print("zapisano " + plik4.split("/")[-1] + " (" + str(len(dane_test)) + " rekordow)")
print("rozklad klas train")
print(dane_train[cel].value_counts())
print("rozklad klas test")
print(dane_test[cel].value_counts())