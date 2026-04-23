import os
import math
import pygad

# zad 2 stop metali

os.makedirs("wykresy", exist_ok=True)


def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)


def fitness_func(ga_instance, solution, solution_idx):
    x = solution[0]
    y = solution[1]
    z = solution[2]
    u = solution[3]
    v = solution[4]
    w = solution[5]

    fit = endurance(x, y, z, u, v, w)
    return fit


def zrob_jedno_uruchomienie(seed=None, zapisz_wykres=False):
    ga_instance = pygad.GA(
        num_generations=80,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=20,
        num_genes=6,
        gene_type=float,
        gene_space={"low": 0.0, "high": 1.0},
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=20,
        random_seed=seed,
    )

    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    if zapisz_wykres:
        fig = ga_instance.plot_fitness(title="zadanie 2 - stop metali - fitness")
        fig.savefig("wykresy/task2_stop_metali_fitness.png")

    return solution, solution_fitness, ga_instance.generations_completed


print("ZADANIE 2 KILKA URUCHOMIEN")

for seed in range(5):
    solution, solution_fitness, pokolenia = zrob_jedno_uruchomienie(
        seed=seed,
        zapisz_wykres=(seed == 0)
    )

    print()
    print("pproba:", seed)
    print("parameters of the best solution :", solution)
    print("fitness value of the best solution =", solution_fitness)
    print("liczba pokolen =", pokolenia)

print()
print("wykres z pierwszego uruchomienia zapisany do: wykresy/task2_stop_metali_fitness.png")