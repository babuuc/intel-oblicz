print("zdecydowalem sie na algorytm aco ktory mozna zastosowac na grafach, poznizej jest przyklad dla 3x3")

# zera oznaczaja sciany a jedynki pola po ktorych mozna chodzic
labirynt = [
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 1]
]

wierzcholki = []
# dodajemy do grafu tylko wolne pola wedlug zalecen w instrukcji
for rzad in range(3):
    for kolumna in range(3):
        if labirynt[rzad][kolumna] == 1:
            wierzcholki.append((rzad, kolumna))

print("wygenerowane wierzcholki to" , wierzcholki)
print("trzeba by wprowadzic nieskonczone odleglosci miedzy punktami bez wspolnej sciany")