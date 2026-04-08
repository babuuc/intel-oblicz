import os
import math
import numpy as np
import pandas as pd

katalog = os.path.dirname(os.path.abspath(__file__))
plik_treningowy = katalog + "/iris_train.csv"
plik_testowy = katalog + "/iris_test.csv"

if os.path.exists(plik_treningowy) == False or os.path.exists(plik_testowy) == False:
    raise Exception("brak iris_train.csv lub iris_test.csv najpierw uruchom zad2.py")

dane_train = pd.read_csv(plik_treningowy)
dane_test = pd.read_csv(plik_testowy)

kolumny = []
kolumny.append(dane_train.columns[0])
kolumny.append(dane_train.columns[1])
kolumny.append(dane_train.columns[2])
kolumny.append(dane_train.columns[3])
kolumna_cel = dane_train.columns[4]

x_tren = dane_train[kolumny].values
y_tren = dane_train[kolumna_cel].values

x_test = dane_test[kolumny].values
y_test = dane_test[kolumna_cel].values


def funkcja_knn(x, k):
    # odleglosc euklidesowa do wszystkich punktow treningowych
    odleglosci = []
    for i in range(len(x_tren)):
        punkt = x_tren[i]
        suma = 0
        for j in range(len(punkt)):
            suma = suma + (punkt[j] - x[j]) ** 2
        wynik_odleglosci = math.sqrt(suma)
        odleglosci.append(wynik_odleglosci)

    kopia_odleglosci = list(odleglosci)
    kopia_odleglosci.sort()
    
    najblizsze_etykiety = []
    for i in range(k):
        najmniejsza = kopia_odleglosci[i]
        indeks = odleglosci.index(najmniejsza)
        najblizsze_etykiety.append(y_tren[indeks])
        # zeby nie wylosowac dwa razy tego samego indeksu
        odleglosci[indeks] = 999999999

    # glosowanie wiekszosciowe przy remisie wybieramy etykiete alfabetycznie
    slownik = {}
    for i in range(len(najblizsze_etykiety)):
        etykieta = najblizsze_etykiety[i]
        if etykieta in slownik:
            slownik[etykieta] = slownik[etykieta] + 1
        else:
            slownik[etykieta] = 1

    najwiecej = 0
    for klucz in slownik:
        if slownik[klucz] > najwiecej:
            najwiecej = slownik[klucz]
            
    zwyciezcy = []
    for klucz in slownik:
        if slownik[klucz] == najwiecej:
            zwyciezcy.append(klucz)
            
    zwyciezcy.sort()
    return zwyciezcy[0]


ka = 5
przewidywania = []
for i in range(len(x_test)):
    wynik = funkcja_knn(x_test[i], ka)
    przewidywania.append(wynik)

dobre = 0
for i in range(len(przewidywania)):
    if przewidywania[i] == y_test[i]:
        dobre = dobre + 1
        
dokladnosc = dobre / len(przewidywania)

dane_wynikowe = dane_test.copy()
dane_wynikowe["predicted"] = przewidywania
plik_wynikowy = katalog + "/iris_test_predictions.csv"
dane_wynikowe.to_csv(plik_wynikowy, index=False)

print("k-nn, k=" + str(ka))
print("dokladnosc: " + str(round(dokladnosc * 100, 2)) + "%")
print("macierz pomylek:")

seria_y = pd.Series(y_test)
seria_y.name = "actual"
seria_pred = pd.Series(przewidywania)
seria_pred.name = "predicted"
print(pd.crosstab(seria_y, seria_pred))

print("zapisano predykcje: " + plik_wynikowy.split("/")[-1])