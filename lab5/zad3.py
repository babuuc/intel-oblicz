# zadanie 3: yolo licznik ptakow
# yolo wykrywa klasy: bird (id=14), airplane (id=4), kite (id=33)
# miniaturki 100x100 sa male dla yolo wiec powiekszymy je przed detekcja

import cv2
import os
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

bird_dir = 'fotki/bird_miniatures/'
pliki = sorted(os.listdir(bird_dir))

# klasy latajacych obiektow w coco
klasy_latajace = {4: 'airplane', 14: 'bird', 33: 'kite'}

print("liczenie ptakow yolo (klasy: bird, airplane, kite)")
print("prog confidence: 0.05 (niski zeby lapac male ptaki)")

for plik in pliki:
    sciezka = os.path.join(bird_dir, plik)
    img = cv2.imread(sciezka)

    # powiekszenie obrazu x4 miniaturki 100x100 sa za male dla yolo
    img_big = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)

    # niski prog bo ptaki na miniaturkach sa male
    results = model(img_big, conf=0.05, verbose=False)
    r = results[0]

    # licz detekcje z klas latajacych
    liczba = 0
    for box in r.boxes:
        klasa_id = int(box.cls)
        if klasa_id in klasy_latajace:
            liczba += 1

    # jesli 0 sprobuj wszystkie detekcje
    if liczba == 0:
        liczba = len(r.boxes)

    print(f"{plik}: {liczba} ptakow (yolo)")

print("\nyolo slabo radzi sobie z miniaturkami 100x100")
print("no i opencv z progowaniem adaptacyjnym daje lepsze wyniki na takich danych")