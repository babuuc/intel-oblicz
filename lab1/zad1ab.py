import math
import datetime

imie = input("Podaj imię: ")
rok = input("Podaj rok urodzenia: ")
miesiac = input("Podaj miesiąc urodzenia: ")
dzien = input("Podaj dzień urodzenia: ")

data_urodzenia = datetime.datetime(int(rok), int(miesiac), int(dzien))
data_dzisiaj = datetime.datetime.now()
dni_zycia = (data_dzisiaj - data_urodzenia).days

fizyczna_fala = math.sin((2 * math.pi * dni_zycia) / 23)
emocjonalna_fala = math.sin((2 * math.pi * dni_zycia) / 28)
intelektualna_fala = math.sin((2 * math.pi * dni_zycia) / 33)

print(f"Witaj {imie}! Dzis twoj {dni_zycia} dzien zycia!")
print(f"Twoj fizyczny wynik biorytmu: {fizyczna_fala}")
print(f"Twoj emocjonalny wynik biorytmu: {emocjonalna_fala}")
print(f"Twoj intelektualny wynik biorytmu: {intelektualna_fala}")

if fizyczna_fala > 0.5:
    print("Twoja kondycja fizyczna jest dzis bardzo dobra!")
elif fizyczna_fala < -0.5:
    fizyczna_jutro = math.sin((2 * math.pi * (dni_zycia + 1)) / 23)
    if fizyczna_jutro > fizyczna_fala:
        print("Twoja kondycja fizyczna jest dzis trudna ale jutro bedzie lepiej!")
    else:
        print("Twoja kondycja fizyczna jest dzis trudna a jutro bedzie gorzej!")

if emocjonalna_fala > 0.5:
    print("Jestes dzisiaj w swietnym nastroju!")
elif emocjonalna_fala < -0.5:
    emocjonalna_jutro = math.sin((2 * math.pi * (dni_zycia + 1)) / 28)
    if emocjonalna_jutro > emocjonalna_fala:
        print("Jestes dzisiaj w slabym nastroju ale jutro bedzie lepiej!")
    else:
        print("Jestes dzisiaj w slabym nastroju a jutro bedzie gorzej!")

if intelektualna_fala > 0.5:
    print("Twoja inteligencja jest dzis na dobrym poziomie!")
elif intelektualna_fala < -0.5:
    intelektualna_jutro = math.sin((2 * math.pi * (dni_zycia + 1)) / 33)
    if intelektualna_jutro > intelektualna_fala:
        print("Twoja inteligencja jest dzis na slabym poziomie ale jutro bedzie lepiej!")
    else:
        print("Twoja inteligencja jest dzis na slabym poziomie a jutro bedzie gorzej!")


# 21 minut i 17 sekund zajelo
