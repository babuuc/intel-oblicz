import math
import random
import matplotlib.pyplot as plt

v = 50
h = 100
g = 9.81

def trajektoria(alfa):
    alfa_rad = math.radians(alfa)
    v_sin = v * math.sin(alfa_rad)
    v_cos = v * math.cos(alfa_rad)

    # czas lotu
    t_hit = (v_sin + math.sqrt(v_sin**2 + 2 * g * h)) / g
    # zasieg
    d = v_cos * t_hit

    return d, t_hit

def rysuj_trajektorie(alfa, t_hit):
    alfa_rad = math.radians(alfa)
    v_sin = v * math.sin(alfa_rad)
    v_cos = v * math.cos(alfa_rad)

    n = 300
    ts = [t_hit * i / (n - 1) for i in range(n)]
    xs = [v_cos * t for t in ts]
    ys = [h + v_sin * t - 0.5 * g * t * t for t in ts]

    plt.figure()
    plt.plot(xs, ys)
    plt.grid(True)
    plt.xlabel("dystans")
    plt.ylabel("wysokosc")
    plt.title("trajektoria pocisku")
    plt.savefig("trajektoria.png", dpi=150, bbox_inches="tight")
    plt.close()


target = random.randint(50, 340)
tolerance = 5
print(f"cel: {target} m")
proby = 0

d = float("inf")
t_hit = 0.0
alfa = 0.0

while abs(d - target) > tolerance:
    alfa = float(input("podaj kat w stopniach: "))
    proby += 1
    d, t_hit = trajektoria(alfa)
    print(f"proba {proby}: zasieg = {d:.2f} m")
    delta = d - target
    if abs(delta) > tolerance:
        if delta < 0:
            print(f"za krotko o {-delta:.2f} m")
        else:
            print(f"za daleko o {delta:.2f} m")

print(f"cel trafiony za proba: {proby}")

rysuj_trajektorie(alfa, t_hit)
print("zapisano wykres: trajektoria.png")