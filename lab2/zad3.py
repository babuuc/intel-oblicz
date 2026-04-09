import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("iris_big.csv")

# bierzemy tylko sepal length i sepal width
X = df[['sepal length (cm)', 'sepal width (cm)']].values
y = df['target_name'].values

# normalizacja min-max
scaler_mm = MinMaxScaler()
X_minmax = scaler_mm.fit_transform(X)

# normalizacja z-score
scaler_z = StandardScaler()
X_zscore = scaler_z.fit_transform(X)

# statystyki
print("=== oryginalne ===")
print("min:", X.min(axis=0))
print("max:", X.max(axis=0))
print("mean:", X.mean(axis=0))
print("std:", X.std(axis=0))

print("\n=== min-max ===")
print("min:", X_minmax.min(axis=0))
print("max:", X_minmax.max(axis=0))
print("mean:", X_minmax.mean(axis=0))
print("std:", X_minmax.std(axis=0))

print("\n=== z-score ===")
print("min:", X_zscore.min(axis=0))
print("max:", X_zscore.max(axis=0))
print("mean:", X_zscore.mean(axis=0))
print("std:", X_zscore.std(axis=0))

# wykresy
gatunki = sorted(set(y))
kolory = ['blue', 'orange', 'green']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

dane = [
    (X, "Original Dataset", "Sepal Length (cm)", "Sepal Width (cm)"),
    (X_zscore, "Z-Core Scaled Dataset", "Sepal Length (cm)", "Sepal Width (cm)"),
    (X_minmax, "Min-Max Normalised Dataset", "Sepal Length (cm)", "Sepal Width (cm)"),
]

for ax, (data, tytul, xlabel, ylabel) in zip(axes, dane):
    for i, gatunek in enumerate(gatunki):
        maska = y == gatunek
        # wyciagamy sam krotki nazwy gatunku
        nazwa = gatunek.replace("Iris-", "").replace("iris-", "")
        ax.scatter(data[maska, 0], data[maska, 1],
                   label=nazwa, color=kolory[i], alpha=0.5, s=10)
    ax.set_title(tytul)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

plt.tight_layout()
plt.savefig("normalizacja.png")
plt.show()
print("zapisano normalizacja.png")