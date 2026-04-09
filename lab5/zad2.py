# zadanie 2: opencv licznik ptakow
# metoda: skala szarosci + adaptacyjny prog + kontury

import cv2
import numpy as np
import os
import math

bird_dir = 'fotki/bird_miniatures/'
output_dir = 'wyniki2'


def zrob_montaz_wykryc(wykrycia, zapis_sciezka):
    if not wykrycia:
        return

    kolumny = 4
    mini_w = 170
    mini_h = 170
    naglowek_h = 45
    podpis_h = 22
    margines = 10

    wiersze = math.ceil(len(wykrycia) / kolumny)
    szerokosc = margines + kolumny * (mini_w + margines)
    wysokosc = naglowek_h + margines + wiersze * (mini_h + podpis_h + margines)

    montaz = np.full((wysokosc, szerokosc, 3), 230, dtype=np.uint8)
    cv2.putText(montaz, 'wyniki opencv - czerwone ramki = wykryte ptaki',
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)

    for i, (podglad, etykieta) in enumerate(wykrycia):
        rzad = i // kolumny
        kol = i % kolumny
        x = margines + kol * (mini_w + margines)
        y = naglowek_h + margines + rzad * (mini_h + podpis_h + margines)

        miniatura = cv2.resize(podglad, (mini_w, mini_h), interpolation=cv2.INTER_AREA)
        montaz[y:y + mini_h, x:x + mini_w] = miniatura
        cv2.putText(montaz, etykieta, (x, y + mini_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (35, 35, 35), 1, cv2.LINE_AA)

    cv2.imwrite(zapis_sciezka, montaz)

def licz_ptaki(sciezka):
    img = cv2.imread(sciezka)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # clahe poprawia kontrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # jakis adaptacyjny prog bo lepszy od globalnego bo tlo jest niejednorodne
    # inv bo ptaki ciemne na jasnym tle wiec odwracamy
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # usun pojedyncze piksele szumu
    costam_do_morpha = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, costam_do_morpha)

    # znajdz kontury
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filtruj po powierzchni ptaki maja 5-200 pikseli
    ptaki = [c for c in contours if 5 <= cv2.contourArea(c) <= 200]

    podglad = img.copy()
    for kontur in ptaki:
        x, y, w, h = cv2.boundingRect(kontur)
        cv2.rectangle(podglad, (x, y), (x + w, y + h), (0, 0, 255), 1)

    return len(ptaki), podglad

# przetwarzanie wszystkich obrazow
pliki = sorted(os.listdir(bird_dir))
os.makedirs(output_dir, exist_ok=True)

linie = []
wykrycia = []

linia_start = "liczenie ptakow na obrazach:"
print(linia_start)
linie.append(linia_start)

for plik in pliki:
    sciezka = os.path.join(bird_dir, plik)
    liczba, podglad = licz_ptaki(sciezka)

    linia = f"{plik}: {liczba} ptakow"
    print(linia)
    linie.append(linia)

    skrot = plik.split('_')[0]
    wykrycia.append((podglad, f"{skrot}: {liczba} ptakow"))

with open(os.path.join(output_dir, 'wyniki.txt'), 'w', encoding='utf-8') as f:
    f.write("\n".join(linie) + "\n")

zrob_montaz_wykryc(wykrycia, os.path.join(output_dir, 'wykrycia.jpg'))