# odpowiedzi do laboratorium inteligencja roju pso i aco

## zadanie 1 pso dla stopu metali
a przyklad z tutoriala
uzyto globalbestpso na sphere z c1 zero piec c2 zero trzy w zero dziewiec
wynik koszt bardzo maly blisko zera
pso dobrze znajduje minimum

b dodanie ograniczen
ustawiono min jeden max dwa
pso respektuje granice znajduje minimum wewnatrz

c zmiana na endurance
granice zero jeden szesc wymiarow
uzyto np zeros i ones

d custom objective
poprawiono endurance na tablice p
dopisano f z petla po roju i minus dla maksimum
zgodne z tutorialem custom objective

e zmiana na maksimum
bez minusa byloby minimum
po minusie najlepszy koszt ujemny okolo minus dwa osm
max endurance okolo dwa osm blisko maksimum teoretycznego trzy

f wykres kosztu
uzyto plot cost history
wykres pokazuje spadek kosztu czyli wzrost endurance
zapisano png w artifacts

## zadanie 2 aco dla komiwojazera
a uruchomienie aco tsp
uruchomiono z siedmioma punktami z pdf
algorytm znalazl sciezke i narysowal

b wieksza liczba punktow
wygenerowano pietnascie losowych punktow zero sto
uruchomiono aco
obliczono dlugosc sciezki okolo czterysta szescset

c modyfikacja parametrow
testowano rozne alpha beta evaporation
wniosek wieksze alpha wiecej eksploatacji feromonu
wieksze beta wiecej eksploracji heurystyki
mniejsza evaporation dluzej szuka globalnie
wieksza liczba mrowek lepsza jakosc ale wolniej

d grid piec na piec
wygenerowano siatke dwadziescia piec punktow
najkrotsza po ludzku to serpentynka wierszami dwiescie czterdziesci
aco znajduje bliska ale nie zawsze idealna bo heurystyka probabilistyczna
w prostym gridzie latwo znalezc krotka

## zadanie 3 strategie roju w labiryncie
wybrano aco
labirynt jako graf wezly komorki krawedzie ruchy bez scian
aco na grafie znajduje najkrotsza sciezke
pso trudniej bo dyskretne rozwiazania
obejscie zaokraglenie i kara za sciane ale ryzyko lokalnego minimum
niepowodzenia jakies moze za maly graf