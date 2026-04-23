import os
import time
import pygad
import numpy as np

# zad 3 labirynt

os.makedirs("wykresy", exist_ok=True)

# 1 = sciana, 0 = wolne pole
# macierz 12x12 razem z ramka scian
maze = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,1,0,0,1],
    [1,1,1,0,0,0,1,0,1,1,0,1],
    [1,0,0,0,1,0,1,0,0,0,0,1],
    [1,0,1,0,1,1,0,0,1,1,0,1],
    [1,0,0,1,1,0,0,0,1,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,1,1],
    [1,0,1,0,0,1,1,0,1,0,0,1],
    [1,0,1,1,1,0,0,0,1,1,0,1],
    [1,0,1,0,1,1,0,1,0,1,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1],
])

start = (1, 1)
exit_pos = (10, 10)
max_kroki = 30

# geny:
# 0 = gora
# 1 = dol
# 2 = lewo
# 3 = prawo
ruchy = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


# ta funkcja przechodzi po chromosomie i sprawdza co sie dzieje w labiryncie
def przejdz_labirynt(solution):
    r = start[0]
    c = start[1]

    odwiedzone = set()
    odwiedzone.add((r, c))

    score = 0
    czy_doszedl = False
    ile_krokow = 0

    for gene in solution:
        dr, dc = ruchy[int(gene)]
        nr = r + dr
        nc = c + dc
        ile_krokow += 1

        # kara za wejscie w sciane
        if maze[nr][nc] == 1:
            score -= 3
        else:
            r = nr
            c = nc
            score += 1

            # premia za nowe pole i kara za krazenie po starym
            if (r, c) in odwiedzone:
                score -= 1
            else:
                score += 2
                odwiedzone.add((r, c))

        # duza premia za dojscie do wyjscia
        if (r, c) == exit_pos:
            score += 1000 + (max_kroki - ile_krokow) * 5
            czy_doszedl = True
            break

    # dodatkowa ocena za bycie blisko wyjscia
    dystans_manhattan = abs(r - exit_pos[0]) + abs(c - exit_pos[1])
    score += 100 - dystans_manhattan * 5

    return score, czy_doszedl, (r, c), ile_krokow


# to jest fitness dla pygada
def fitness_func(ga_instance, solution, solution_idx):
    score, czy_doszedl, koniec, ile_krokow = przejdz_labirynt(solution)
    return score


# zamiana chromosomu na napis z ruchami
def zamien_na_napis(solution):
    tekst = ""
    mapa = {0: "U", 1: "D", 2: "L", 3: "R"}

    for x in solution:
        tekst += mapa[int(x)]

    return tekst


# jedno uruchomienie
def zrob_jedno_uruchomienie(seed=None, zapisz_wykres=False):
    ga_instance = pygad.GA(
        num_generations=800,
        num_parents_mating=40,
        fitness_func=fitness_func,
        sol_per_pop=80,
        num_genes=max_kroki,
        gene_type=int,
        gene_space=[0, 1, 2, 3],
        parent_selection_type="sss",
        keep_parents=4,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=4,
        stop_criteria=["reach_1000"],
        random_seed=seed,
    )

    start_time = time.time()
    ga_instance.run()
    end_time = time.time()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    score, czy_doszedl, koniec, ile_krokow = przejdz_labirynt(solution)

    if zapisz_wykres:
        fig = ga_instance.plot_fitness(title="zadanie 3 - labirynt - fitness")
        fig.savefig("wykresy/task3_labirynt_fitness.png")

    return {
        "solution": solution,
        "fitness": solution_fitness,
        "czas": end_time - start_time,
        "czy_doszedl": czy_doszedl,
        "koniec": koniec,
        "ile_krokow": ile_krokow,
        "ruchy": zamien_na_napis(solution),
        "pokolenia": ga_instance.generations_completed,
    }


print("ZADANIE 3 JEDNO URUCHOMIENIE")

wynik = zrob_jedno_uruchomienie(seed=0, zapisz_wykres=True)
print("najlepszy chromosom:", wynik["solution"])
print("fitness:", wynik["fitness"])
print("czy doszedl do wyjscia:", wynik["czy_doszedl"])
print("pozycja koncowa:", wynik["koniec"])
print("liczba wykorzystanych krokow:", wynik["ile_krokow"])
print("ruchy:", wynik["ruchy"])
print("liczba pokolen:", wynik["pokolenia"])
print("czas:", wynik["czas"])
print("wykres zapisany do: wykresy/task3_labirynt_fitness.png")

print()
print("ZADANIE 3 10 URUCHOMIEN")

ile_udanych = 0
czasy_udanych = []

for seed in range(10):
    wynik = zrob_jedno_uruchomienie(seed=seed, zapisz_wykres=False)

    if wynik["czy_doszedl"]:
        ile_udanych += 1
        czasy_udanych.append(wynik["czas"])

    print(
        "proba =", seed,
        "| sukces =", wynik["czy_doszedl"],
        "| fitness =", wynik["fitness"],
        "| koniec =", wynik["koniec"],
        "| kroki =", wynik["ile_krokow"],
        "| czas =", wynik["czas"],
    )

print()
print("udanych prob:", ile_udanych, "/ 10")

if len(czasy_udanych) > 0:
    sredni_czas = sum(czasy_udanych) / len(czasy_udanych)
    print("sredni czas z udanych prob:", sredni_czas)
else:
    print("brak udanych prob nie da sie policzyc sredniego czasu")