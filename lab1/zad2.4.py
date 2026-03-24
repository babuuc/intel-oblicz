import math
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TrebuchetConfig:
    v0: float = 50.0   # m/s (stałe w zadaniu)
    h: float = 100.0   # m   (stałe w zadaniu)
    g: float = 9.81    # m/s^2
    target_min: int = 50
    target_max: int = 340
    tolerance: float = 5.0
    plot_points: int = 600
    output_png: str = "trajektoria.png"


def distance_for_angle_deg(angle_deg: float, cfg: TrebuchetConfig) -> float:
    """
    Zasięg d z aneksu:
    d = (v0*sin(a) + sqrt(v0^2*sin^2(a) + 2*g*h)) * (v0*cos(a))/g
    """
    a = math.radians(angle_deg)
    sin_a = math.sin(a)
    cos_a = math.cos(a)

    if cos_a == 0.0:
        return 0.0  # strzał pionowo w górę -> brak zasięgu poziomego

    under_sqrt = (cfg.v0 ** 2) * (sin_a ** 2) + 2.0 * cfg.g * cfg.h
    return (cfg.v0 * sin_a + math.sqrt(under_sqrt)) * (cfg.v0 * cos_a) / cfg.g


def trajectory_y(x: float, angle_deg: float, cfg: TrebuchetConfig) -> float:
    """
    Trajektoria y(x) z aneksu:
    y = -(g/(2*v0^2*cos^2(a))) * x^2 + (sin(a)/cos(a))*x + h
    """
    a = math.radians(angle_deg)
    sin_a = math.sin(a)
    cos_a = math.cos(a)

    # Unikamy dzielenia przez 0 (kąt 90°)
    if abs(cos_a) < 1e-12:
        return float("nan")

    return -(cfg.g / (2.0 * (cfg.v0 ** 2) * (cos_a ** 2))) * (x ** 2) + (sin_a / cos_a) * x + cfg.h


def plot_trajectory(angle_deg: float, distance: float, cfg: TrebuchetConfig) -> None:
    xs = [distance * i / (cfg.plot_points - 1) for i in range(cfg.plot_points)]
    ys = [trajectory_y(x, angle_deg, cfg) for x in xs]

    plt.figure()
    plt.plot(xs, ys)  # nie ustawiamy kolorów na sztywno (domyślnie będzie niebieski)
    plt.grid(True)
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.title("Projectile Motion for the Trebuchet")
    plt.savefig(cfg.output_png, dpi=150, bbox_inches="tight")
    plt.close()


def read_angle() -> float:
    raw = input("Podaj kąt α w stopniach (0 < α < 90): ").strip().replace(",", ".")
    angle = float(raw)
    if not (0.0 < angle < 90.0):
        raise ValueError("Kąt musi być w zakresie (0, 90).")
    return angle


def main() -> None:
    cfg = TrebuchetConfig()
    target = random.randint(cfg.target_min, cfg.target_max)
    print(f"Cel znajduje się w odległości: {target} m")
    print(f"Trafienie, jeśli pocisk spadnie w zakresie [{target - cfg.tolerance}, {target + cfg.tolerance}] m")

    attempts = 0
    winning_angle = None
    winning_distance = None

    while True:
        attempts += 1
        try:
            angle = read_angle()
        except ValueError as e:
            print(f"Błąd: {e} Spróbuj jeszcze raz.")
            attempts -= 1  # nie liczmy błędnej próby jako strzału
            continue

        d = distance_for_angle_deg(angle, cfg)
        print(f"Próba #{attempts}: dla α={angle:.2f}° pocisk doleci na d={d:.2f} m")

        if abs(d - target) <= cfg.tolerance:
            print(f"Cel trafiony! Liczba prób: {attempts}")
            winning_angle = angle
            winning_distance = d
            break

        if d < target:
            print(f"Chybiony: za krótko o {target - d:.2f} m.")
        else:
            print(f"Chybiony: za daleko o {d - target:.2f} m.")

    # Część 3 zadania: rysunek trajektorii ostatniego (trafionego) strzału
    plot_trajectory(winning_angle, winning_distance, cfg)
    print(f"Zapisano wykres trajektorii do pliku: {cfg.output_png}")


if __name__ == "__main__":
    main()