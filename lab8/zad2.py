import matplotlib.pyplot as plt
import random
from aco import AntColony

print("start zadania drugiego aco dla komiwojazera")

print("a uruchomienie aco tsp z siedmioma punktami")
COORDS = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
)
colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2, 
                    pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
                    iterations=300)
optimal_nodes = colony.get_path()
print("znaleziono sciezke dla siedmiu punktow")

print("b wieksza liczba punktow pietnascie losowych od zero do sto")
random.seed(42)
coords15 = [(random.randint(0,100), random.randint(0,100)) for _ in range(15)]
print("punkty pietnascie wygenerowane")

def dlugosc_sciezki(path):
    total = 0.0
    for i in range(len(path)-1):
        dx = path[i][0] - path[i+1][0]
        dy = path[i][1] - path[i+1][1]
        total += (dx*dx + dy*dy)**0.5
    dx = path[-1][0] - path[0][0]
    dy = path[-1][1] - path[0][1]
    total += (dx*dx + dy*dy)**0.5
    return total

colony15 = AntColony(coords15, ant_count=50, alpha=0.6, beta=1.5, pheromone_evaporation_rate=0.35, pheromone_constant=800.0, iterations=30)
path15 = colony15.get_path()
len15 = dlugosc_sciezki(path15)
print("dlugosc dla pietnastu punktow " + str(len15))

print("c eksperyment z innymi parametrami")
colony15b = AntColony(coords15, ant_count=100, alpha=0.8, beta=2.0, pheromone_evaporation_rate=0.25, pheromone_constant=1500.0, iterations=50)
path15b = colony15b.get_path()
len15b = dlugosc_sciezki(path15b)
print("dlugosc z nowymi parametrami " + str(len15b))
print("zmiana parametrow wplywa na dlugosc sciezki wieksze alpha wiecej feromonu wiecej eksploatacji")

print("d siatka piec na piec grid")
coords_grid = []
for i in range(5):
    for j in range(5):
        coords_grid.append((i*10, j*10))
print("siatka dwadziescia piec punktow gotowa")
colony_grid = AntColony(coords_grid, ant_count=50, alpha=0.5, beta=1.0, pheromone_evaporation_rate=0.3, pheromone_constant=1000.0, iterations=30)
path_grid = colony_grid.get_path()
len_grid = dlugosc_sciezki(path_grid)
print("dlugosc w grid " + str(len_grid))
print("najkrotsza po ludzku to serpentynka wierszami dwiescie czterdziesci jednostek")
print("aco znajduje bliska wartosc ale nie zawsze idealna")

print("zadanie drugie gotowe")