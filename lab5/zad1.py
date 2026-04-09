# zadanie 1: detekcja obiektow yolo
# yolov8n nano model trenowany na coco (80 klas)
# architektura: backbone CSPDarknet + neck FPN + head detekcyjny
# zbior treningowy: coco dataset

import json
import cv2
from ultralytics import YOLO
from collections import defaultdict

model = YOLO('yolov8n.pt')  # ma pobrac wagi z neta przy pierwszym odpaleniu

# a) info o modelu
print("model: yolov8n")
print("liczba klas:", len(model.names))
print("przykladowe klasy:", [model.names[i] for i in range(10)])
print("zbior treningowy: coco (80 klas)")

# b) c) detekcja na zdjeciu z rozymi progami confidence
zdjecie = 'office_yolo.png'
progi = [0.1, 0.3, 0.5, 0.7]
output_dir = 'wyniki1'

for prog in progi:
    results = model(zdjecie, conf=prog, verbose=False)
    r = results[0]

    detekcje = []
    for box in r.boxes:
        detekcje.append({
            'klasa_id': int(box.cls),
            'klasa_nazwa': model.names[int(box.cls)],
            'confidence': round(float(box.conf), 4),
            'bbox_xyxy': [round(float(v), 1) for v in box.xyxy[0].tolist()]
        })

    # zapis json
    with open(f'{output_dir}/detekcje_zdjecie_conf{prog}.json', 'w') as f:
        json.dump({'prog_confidence': prog, 'liczba_detekcji': len(detekcje), 'detekcje': detekcje}, f, indent=2)
    print(f"conf={prog}: wykryto {len(detekcje)} obiektow -> detekcje_zdjecie_conf{prog}.json")

    # zapis obrazu z boxami tymi
    img_boxes = r.plot()
    cv2.imwrite(f'{output_dir}/wynik_zdjecie_conf{prog}.jpg', img_boxes)

# d) detekcja na filmach klatka po klatce
for nazwa_pliku in ['office_yolo.mp4', 'street_yolo.mp4']:
    for prog in [0.3, 0.5]:
        cap = cv2.VideoCapture(nazwa_pliku)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        nazwa_bez = nazwa_pliku.replace('.mp4', '')
        out = cv2.VideoWriter(
            f'{output_dir}/wynik_{nazwa_bez}_conf{prog}.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
        )

        wyniki_film = []
        statystyki = defaultdict(int)
        numer_klatki = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=prog, verbose=False)
            r = results[0]

            klatka_detekcje = []
            for box in r.boxes:
                klasa = model.names[int(box.cls)]
                statystyki[klasa] += 1
                klatka_detekcje.append({
                    'klasa': klasa,
                    'confidence': round(float(box.conf), 4),
                    'bbox': [round(float(v), 1) for v in box.xyxy[0].tolist()]
                })

            wyniki_film.append({
                'klatka': numer_klatki,
                'czas_s': round(numer_klatki / fps, 3),
                'detekcje': klatka_detekcje
            })

            out.write(r.plot())
            numer_klatki += 1

        cap.release()
        out.release()

        # zapis json z wszystkimi klatkami
        nazwa_json = f'{output_dir}/detekcje_{nazwa_bez}_conf{prog}.json'
        with open(nazwa_json, 'w') as f:
            json.dump({'film': nazwa_pliku, 'prog': prog, 'klatki': wyniki_film}, f, indent=2)

        # e) staty
        print(f"\n{nazwa_pliku} conf={prog}: {numer_klatki} klatek")
        print("statystyki klas:")
        for klasa, ile in sorted(statystyki.items(), key=lambda x: -x[1]):
            print(f"  {klasa}: {ile}")

        with open(f'{output_dir}/statystyki_{nazwa_bez}_conf{prog}.txt', 'w') as f:
            f.write(f"film: {nazwa_pliku}, conf: {prog}\n")
            for klasa, ile in sorted(statystyki.items(), key=lambda x: -x[1]):
                f.write(f"{klasa}: {ile}\n")

print("\ngotowe")