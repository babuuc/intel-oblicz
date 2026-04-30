import numpy as np
import math
import matplotlib.pyplot as plt
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history

print("start zadania pierwszego pso dla stopu metali")

print("a przyklad z tutoriala basic optimization na sphere")
options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
cost, pos = optimizer.optimize(fx.sphere, iters=200, verbose=False)
print("sphere zoptymalizowane koszt " + str(cost) + " pozycja " + str(pos))
print("pso znajduje minimum blisko zera")

print("b dodanie ograniczen min jeden max dwa")
x_max = [2, 2]
x_min = [1, 1]
my_bounds = (x_min, x_max)
optimizer2 = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=my_bounds)
cost2, pos2 = optimizer2.optimize(fx.sphere, iters=200, verbose=False)
print("z ograniczeniami koszt " + str(cost2) + " pozycja " + str(pos2))
print("minimum w obrebie granic")

print("c zmiana na endurance szesc wymiarow granice zero jeden")
def endurance(p):
    x = p[0]
    y = p[1]
    z = p[2]
    u = p[3]
    v = p[4]
    w = p[5]
    val = math.exp(-2 * (y - math.sin(x))**2) + math.sin(z * u) + math.cos(v * w)
    return val

print("d funkcja f dla calego roju z minusem dla maksimum")
def f(x):
    n = x.shape[0]
    j = np.zeros(n)
    for i in range(n):
        j[i] = -1 * endurance(x[i])
    return j

print("e optymalizacja z bounds zero jeden")
x_max = np.ones(6)
x_min = np.zeros(6)
bounds = (x_min, x_max)
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=6, options=options, bounds=bounds)
cost, pos = optimizer.optimize(f, iters=100, verbose=False)
print("optymalizacja gotowa")
print("najlepszy koszt " + str(cost))
print("najlepsza pozycja " + str(pos))
print("max endurance to " + str(-cost))

print("f wykres kosztu")
cost_history = optimizer.cost_history
plot_cost_history(cost_history)
plt.savefig("cost_history_endurance.png")
print("wykres zapisany jako cost_history_endurance.png")

print("zadanie pierwsze gotowe")