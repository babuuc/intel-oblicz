import os
import time
import pygad
import numpy as np

# zad 1 prolem plecakowy

# kazdy przedmiot ma nazwe, wartosc i wage
items = [
    {"nazwa": "zegar", "wartosc": 100, "waga": 7},
    {"nazwa": "obraz-pejzaz", "wartosc": 300, "waga": 7},
    {"nazwa": "obraz-portret", "wartosc": 200, "waga": 6},
    {"nazwa": "radio", "wartosc": 40, "waga": 2},
    {"nazwa": "laptop", "wartosc": 500, "waga": 5},
    {"nazwa": "lampka nocna", "wartosc": 70, "waga": 6},
    {"nazwa": "srebrne sztucce", "wartosc": 100, "waga": 1},
    {"nazwa": "porcelana", "wartosc": 250, "waga": 3},
    {"nazwa": "figura z brazu", "wartosc": 300, "waga": 10},
    {"nazwa": "skorzana torebka", "wartosc": 280, "waga": 3},
    {"nazwa": "odkurzacz", "wartosc": 300, "waga": 15},
]

limit_wagi = 25
najlepsza_wartosc_z_tresci = 1630

wartosci = np.array([item["wartosc"] for item in items])
wagi = np.array([item["waga"] for item in items])

# katalog na wykresy
os.makedirs("wykresy", exist_ok=True)


def fitness_func(ga_instance, solution, solution_idx):
    # solution to np. [0,1,1,0,1,0,1,1,0,1,0]
    # 1 znaczy: bierzemy przedmiot
    suma_wartosci = np.sum(solution * wartosci)
    suma_wagi = np.sum(solution * wagi)

    # jak przekraczamy limit to dajemy slaba ocene
    if suma_wagi > limit_wagi:
        return 0

    # w innym przypadku fitness = laczna wartosc
    return suma_wartosci


def policz_co_wybrano(solution):
    wybrane = []
    suma_wartosci = 0
    suma_wagi = 0

    for i in range(len(solution)):
        if int(solution[i]) == 1:
            wybrane.append(items[i]["nazwa"])
            suma_wartosci += items[i]["wartosc"]
            suma_wagi += items[i]["waga"]

    return wybrane, suma_wartosci, suma_wagi


def zrob_jedno_uruchomienie(seed=None, zapisz_wykres=False):
    ga_instance = pygad.GA(
        num_generations=300,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=20,
        num_genes=len(items),
        gene_type=int,
        gene_space=[0, 1],
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10,
        stop_criteria=["reach_1630"],
        random_seed=seed,
    )

    start = time.time()
    ga_instance.run()
    end = time.time()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    wybrane, suma_wartosci, suma_wagi = policz_co_wybrano(solution)

    if zapisz_wykres:
        fig = ga_instance.plot_fitness(title="zadanie 1 - plecak - fitness")
        fig.savefig("wykresy/task1_knapsack_fitness.png")

    return {
        "solution": solution,
        "fitness": solution_fitness,
        "czas": end - start,
        "wybrane": wybrane,
        "suma_wartosci": suma_wartosci,
        "suma_wagi": suma_wagi,
        "pokolenia": ga_instance.generations_completed,
    }


# run 1 raz
print("ZADANIE 1 JEDNO URUCHOMIENIE")

wynik = zrob_jedno_uruchomienie(seed=0, zapisz_wykres=True)

print("Najlepszy chromosom:", wynik["solution"])
print("Fitness:", wynik["fitness"])
print("Wybrane przedmioty:", wynik["wybrane"])
print("Laczna wartosc:", wynik["suma_wartosci"])
print("Laczna waga:", wynik["suma_wagi"])
print("Liczba pokolen:", wynik["pokolenia"])
print("Czas:", wynik["czas"])
print("Wykres zapisany do: wykresy/task1_knapsack_fitness.png")

# test 10 runow ga
print()
print("ZADANIE 1 10 URUCHOMIEN")

ile_udanych = 0
czasy_udanych = []

for seed in range(10):
    wynik = zrob_jedno_uruchomienie(seed=seed, zapisz_wykres=False)
    sukces = wynik["fitness"] == najlepsza_wartosc_z_tresci

    if sukces:
        ile_udanych += 1
        czasy_udanych.append(wynik["czas"])

    print(
        "proba =", seed,
        "| fitness =", wynik["fitness"],
        "| wartosc =", wynik["suma_wartosci"],
        "| waga =", wynik["suma_wagi"],
        "| sukces =", sukces,
        "| czas =", wynik["czas"],
    )

skutecznosc = (ile_udanych / 10) * 100
print()
print("skutecznosc z 10 uruchomien:", skutecznosc, "%")

if len(czasy_udanych) > 0:
    sredni_czas = sum(czasy_udanych) / len(czasy_udanych)
    print("sredni czas z udanych prob:", sredni_czas)
else:
    print("brak udanych prob nie da sie policzyc sredniego czasu")