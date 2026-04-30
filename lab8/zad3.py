import random
print("start zadania trzeciego strategie roju w labiryncie")

print("wybralem podejscie aco bo labirynt latwo zakodowac jako graf")
print("wezly to komorki korytarzy krawedzie to mozliwe ruchy gora dol lewo prawo bez scian")
print("standardowa paczka aco jest dla tsp pelnego grafu wiec zrobilem custom prosta symulacje aco na siatce")

print("definiuje prosty labirynt piec na piec zero wolne jeden sciana")
maze = [
[0, 0, 1, 0, 0],
[1, 0, 0, 0, 1],
[0, 0, 1, 0, 0],
[0, 1, 0, 0, 0],
[0, 0, 0, 1, 0]
]
start = (0, 0)
end = (4, 4)
print("start " + str(start) + " koniec " + str(end))

print("inicjalizuje feromon na kazdej komorce")
pheromone = [[1.0 for _ in range(5)] for _ in range(5)]
random.seed(42)

print("uruchamiam symulacje aco piecdziesiat iteracji po dziesiec mrowek")
best_path = None
best_len = 999
for it in range(50):
    for ant in range(10):
        path = [start]
        x, y = start
        visited = set([start])
        for step in range(30):
            moves = []
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < 5 and 0 <= ny < 5 and maze[nx][ny] == 0 and (nx, ny) not in visited:
                    moves.append((nx, ny, pheromone[nx][ny]))
            if not moves:
                break
            total = sum(m[2] for m in moves)
            if total <= 0:
                break
            r = random.uniform(0, total)
            cum = 0.0
            chosen = None
            for nx, ny, p in moves:
                cum += p
                if r <= cum:
                    chosen = (nx, ny)
                    break
            if chosen is None:
                break
            x, y = chosen
            path.append((x, y))
            visited.add((x, y))
            if (x, y) == end:
                length = len(path)
                if length < best_len:
                    best_len = length
                    best_path = path[:]
                for px, py in path:
                    pheromone[px][py] += 2.0 / length
                break

print("symulacja zakonczona")
if best_path:
    print("znaleziono sciezke dlugosc " + str(best_len))
    print("sciezka " + str(best_path))
else:
    print("nie znaleziono sciezki w limitach")
    print("przyczyna mozliwe za male wzmocnienie feromonu lub za malo iteracji lub labirynt trudny")

print("zadanie trzecie gotowe")