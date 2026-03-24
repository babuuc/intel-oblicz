import datetime
import math

def oblicz_biorytmy(t):
    """Oblicza wartości trzech biorytmów dla danego dnia t."""
    fizyczny = math.sin((2 * math.pi / 23) * t)
    emocjonalny = math.sin((2 * math.pi / 28) * t)
    intelektualny = math.sin((2 * math.pi / 33) * t)
    return fizyczny, emocjonalny, intelektualny

def analizuj_biorytm(nazwa, dzis, jutro):
    """Analizuje biorytm, wyświetla komunikaty i pociesza w razie potrzeby."""
    print(f"--- Biorytm {nazwa.capitalize()} ---")
    print(f"Wartość dzisiaj: {dzis:.2f}")
    
    if dzis > 0.5:
        print(f"🌟 Super! Twój biorytm {nazwa} jest dzisiaj na świetnym poziomie! Wykorzystaj ten czas na maksa.")
    elif dzis < -0.5:
        print(f"🌧️ Twój biorytm {nazwa} jest dzisiaj słaby. Głowa do góry, każdy ma prawo do gorszego dnia! Odpocznij i nie bądź dla siebie zbyt surowy.")
        
        # Sprawdzanie trendu na jutro
        if jutro > dzis:
            print(f"➡️ Małe pocieszenie: Jutro będzie już zauważalnie lepiej! (wartość wzrośnie do {jutro:.2f}).")
        else:
            print(f"➡️ Uwaga: Jutro ta fala nadal będzie w tendencji spadkowej (wartość wyniesie {jutro:.2f}). Zadbaj o siebie podwójnie i po prostu przetrwaj ten dołek!")
    else:
        print(f"⚖️ Twój biorytm {nazwa} jest na średnim, neutralnym poziomie. Zwykły, stabilny dzień.")
    print("") # Pusta linia dla lepszej czytelności

# --- Główny program ---
print("Witaj w Kalkulatorze Biorytmów!")
imie = input("Podaj swoje imię: ")

# Pobieranie daty od użytkownika
try:
    rok = int(input("Podaj rok urodzenia (np. 1990): "))
    miesiac = int(input("Podaj miesiąc urodzenia (1-12): "))
    dzien = int(input("Podaj dzień urodzenia (1-31): "))
    
    data_urodzenia = datetime.date(rok, miesiac, dzien)
    dzisiaj = datetime.date.today()
    
    # Obliczanie ilości przeżytych dni (t)
    t = (dzisiaj - data_urodzenia).days
    
    if t < 0:
        print("Błąd: Data urodzenia jest w przyszłości!")
    else:
        print(f"\n======================================")
        print(f"Witaj, {imie}!")
        print(f"Dzisiejsza data to: {dzisiaj}")
        print(f"To dokładnie twój {t}. dzień życia na Ziemi.")
        print(f"======================================\n")
        
        # Obliczanie biorytmów na dziś (t) i jutro (t+1)
        fiz_dzis, emo_dzis, int_dzis = oblicz_biorytmy(t)
        fiz_jutro, emo_jutro, int_jutro = oblicz_biorytmy(t + 1)
        
        # Analiza i wypisywanie wyników
        analizuj_biorytm("fizyczny", fiz_dzis, fiz_jutro)
        analizuj_biorytm("emocjonalny", emo_dzis, emo_jutro)
        analizuj_biorytm("intelektualny", int_dzis, int_jutro)

except ValueError:
    print("Błąd: Wprowadzono nieprawidłowe dane (upewnij się, że wpisujesz liczby całkowite dla daty).")

# zajelo to 3 min i 10 sekund gemini 3.1 pro