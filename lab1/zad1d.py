#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import datetime
from typing import Tuple


CYCLES = {
    "fizyczna": 23,
    "emocjonalna": 28,
    "intelektualna": 33,
}


def read_int(prompt: str, min_value: int = None, max_value: int = None) -> int:
    """Czyta int od użytkownika z walidacją zakresu."""
    while True:
        try:
            s = input(prompt).strip()
            value = int(s)
            if min_value is not None and value < min_value:
                print(f"Wprowadź wartość >= {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"Wprowadź wartość <= {max_value}.")
                continue
            return value
        except ValueError:
            print("Nieprawidłowa liczba. Spróbuj jeszcze raz.")


def days_lived(birth: datetime.date, today: datetime.date) -> int:
    return (today - birth).days


def biorhythm_value(days: int, cycle: int) -> float:
    """Zwraca wartość fali w przedziale [-1, 1]."""
    return math.sin((2 * math.pi * days) / cycle)


def interpret(value: float, tomorrow_value: float) -> str:
    """Tworzy tekstową interpretację w oparciu o wartość fali i wartość jutro."""
    if value > 0.5:
        return "bardzo dobra"
    if value < -0.5:
        if tomorrow_value > value:
            return "dzisiaj trudniej — ale jutro będzie lepiej"
        else:
            return "dzisiaj trudniej — jutro będzie gorzej"
    return "stabilna / umiarkowana"


def format_val(v: float) -> str:
    """Formatowanie wartości fali do 3 miejsc po przecinku i procentu (opcjonalnie)."""
    return f"{v:.3f}"


def main():
    try:
        imie = input("Podaj imię: ").strip() or "Użytkowniku"

        rok = read_int("Podaj rok urodzenia (np. 1987): ", min_value=1, max_value=9999)
        miesiac = read_int("Podaj miesiąc urodzenia (1-12): ", min_value=1, max_value=12)
        dzien = read_int("Podaj dzień urodzenia (1-31): ", min_value=1, max_value=31)

        try:
            data_urodzenia = datetime.date(rok, miesiac, dzien)
        except ValueError:
            print("Nieprawidłowa data (np. 30 lutego). Uruchom program ponownie i podaj poprawną datę.")
            return

        data_dzisiaj = datetime.date.today()
        dni = days_lived(data_urodzenia, data_dzisiaj)

        if dni < 0:
            print("Data urodzenia jest w przyszłości — sprawdź datę i spróbuj ponownie.")
            return

        print(f"\nWitaj {imie}! Dziś masz {dni} dni życia (ur. {data_urodzenia.isoformat()}).\n")

        for nazwa, cykl in CYCLES.items():
            today_val = biorhythm_value(dni, cykl)
            tomorrow_val = biorhythm_value(dni + 1, cykl)
            status = interpret(today_val, tomorrow_val)
            print(f"{nazwa.capitalize():12}: {format_val(today_val):>7}  →  {status}")

        # (opcjonalnie) pokaż wartości z większą ilością informacji:
        print("\nSzczegóły (wartości od -1 do 1):")
        for nazwa, cykl in CYCLES.items():
            v = biorhythm_value(dni, cykl)
            print(f" - {nazwa.capitalize():12}: {format_val(v)}")

    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika. Do widzenia.")


if __name__ == "__main__":
    main()