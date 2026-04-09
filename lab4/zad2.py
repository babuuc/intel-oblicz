import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

torch.manual_seed(13)

df = pd.read_csv("iris_big.csv")

# kodowanie klas tekstowych na liczby: setosa=0, versicolor=1, virginica=2
le = LabelEncoder()
X = df.iloc[:, 0:4].values.astype("float32")
y = le.fit_transform(df.iloc[:, 4].values)

# podzial 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=13, stratify=y
)

# normalizacja inputow (mean=0, std=1)
# fit tylko na train a test tylko transform
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# konwersja na tensory pytorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# dataloadery z batch size 32
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=32)

# b) model sieci neuronowej
# topologia: 4 wejscia -> 16 neuronow (relu) -> 8 neuronow (relu) -> 3 wyjscia
# 3 wyjscia bo 3 gatunki irysow
# na koncu nie dajemy softmax, bo CrossEntropyLoss ogarnia to sama
class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.siec = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.siec(x)

model = IrisNet()
strata_fn = nn.CrossEntropyLoss()
optymalizator = torch.optim.Adam(model.parameters(), lr=0.01)

# c) trenowanie
EPOKI = 100

historia_train_loss = []
historia_test_loss  = []
historia_train_acc  = []
historia_test_acc   = []

for epoka in range(EPOKI):
    model.train()
    suma_loss = 0
    poprawne = 0
    lacznie = 0

    for X_batch, y_batch in train_loader:
        optymalizator.zero_grad()
        wyj = model(X_batch)
        loss = strata_fn(wyj, y_batch)
        loss.backward()
        optymalizator.step()

        suma_loss += loss.item() * len(y_batch)
        poprawne  += (wyj.argmax(1) == y_batch).sum().item()
        lacznie   += len(y_batch)

    historia_train_loss.append(suma_loss / lacznie)
    historia_train_acc.append(poprawne / lacznie)

    # sprawdzanie na test
    model.eval()
    suma_loss = 0
    poprawne = 0
    lacznie = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            wyj = model(X_batch)
            loss = strata_fn(wyj, y_batch)

            suma_loss += loss.item() * len(y_batch)
            poprawne  += (wyj.argmax(1) == y_batch).sum().item()
            lacznie   += len(y_batch)

    historia_test_loss.append(suma_loss / lacznie)
    historia_test_acc.append(poprawne / lacznie)

    if (epoka + 1) % 20 == 0:
        print(f"epoka {epoka+1}/{EPOKI}  train_loss={historia_train_loss[-1]:.4f}  test_acc={historia_test_acc[-1]*100:.2f}%")

# d) krzywe uczenia
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(historia_train_loss, label="trening")
ax1.plot(historia_test_loss,  label="walidacja/test")
ax1.set_title("loss")
ax1.set_xlabel("epoka")
ax1.set_ylabel("crossentropy loss")
ax1.legend()

ax2.plot([a * 100 for a in historia_train_acc], label="trening")
ax2.plot([a * 100 for a in historia_test_acc],  label="walidacja/test")
ax2.set_title("accuracy")
ax2.set_xlabel("epoka")
ax2.set_ylabel("accuracy %")
ax2.legend()

plt.tight_layout()
plt.savefig("zad2_krzywe.png")
plt.close()
print("zapisano zad2_krzywe.png")

# e) statystyki koncowe
model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(1).numpy()

acc = (preds == y_test).mean() * 100
cm = confusion_matrix(y_test, preds)

print(f"\nkoncowa dokladnosc na zbiorze testowym: {acc:.2f}%")
print("macierz bledow (setosa / versicolor / virginica):")
print(cm)

# f) interpretacja
# siec ma 2 warstwy ukryte z relu i dobrze radzi sobie z tym datasetem
# normalizacja danych pomaga w uczeniu
# crossentropy jest dobra strata dla wieloklasowej klasyfikacji