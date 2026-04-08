import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("diagnosis.csv")

X = df.iloc[:, 0:3].values.astype("float32")
y = df.iloc[:, 3].values.astype("int64")

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=13)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=32)

# siec: 3 wejscia -> 16 -> 8 -> 2 wyjscia
class DiagNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.siec = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.siec(x)

model = DiagNet()
strata_fn = nn.CrossEntropyLoss()
optymalizator = torch.optim.Adam(model.parameters(), lr=0.01)

EPOKI = 100

historia_train_loss = []
historia_test_loss  = []
historia_train_acc  = []
historia_test_acc   = []

for epoka in range(EPOKI):
    model.train()
    suma_loss = 0; poprawne = 0; lacznie = 0
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

    model.eval()
    suma_loss = 0; poprawne = 0; lacznie = 0
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

# krzywe uczenia
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(historia_train_loss, label="trening")
ax1.plot(historia_test_loss,  label="walidacja")
ax1.set_title("loss"); ax1.set_xlabel("epoka"); ax1.legend()

ax2.plot([a*100 for a in historia_train_acc], label="trening")
ax2.plot([a*100 for a in historia_test_acc],  label="walidacja")
ax2.set_title("accuracy"); ax2.set_xlabel("epoka"); ax2.set_ylabel("%"); ax2.legend()
plt.tight_layout()
plt.savefig("zad3_krzywe.png")
plt.close()
print("zapisano zad3_krzywe.png")

# statystyki koncowe
model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(1).numpy()

acc  = accuracy_score(y_test, preds) * 100
prec = precision_score(y_test, preds) * 100
rec  = recall_score(y_test, preds) * 100
cm   = confusion_matrix(y_test, preds)

print(f"\naccuracy:  {acc:.2f}%")
print(f"precision: {prec:.2f}%")
print(f"recall:    {rec:.2f}%")
print("macierz bledow:")
print(cm)

# heatmapa seaborn
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['zdrowy', 'chory'],
            yticklabels=['zdrowy', 'chory'])
plt.title("macierz bledow - diagnoza mlp")
plt.ylabel("prawdziwa klasa")
plt.xlabel("przewidywana klasa")
plt.tight_layout()
plt.savefig("zad3_cm.png")
plt.close()
print("zapisano zad3_cm.png")

# interpretacja:
# precision: round(prec, 1) % - tyle procent z przewidzianych chorych to faktycznie chorzy
# recall: round(rec, 1) % - tyle procent faktycznie chorych zostalo wykrytych
# recall wazniejszy w medycynie - lepiej blednie powiedziec ze ktos jest chory niz przeoczyc chorobe