import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("iris_big.csv")

# oddzielamy dane numeryczne od gatunkow
num_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
X = df[num_cols].values
y = df['target_name'].values

# robimy pca na 4 skladowe (tyle ile kolumn)
pca = PCA(n_components=4)
pca.fit(X)

print("=== wariancja dla kazdej skladowej ===")
for i, v in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {v:.4f} ({v*100:.2f}%)")

print("\n=== skumulowana wariancja ===")
suma = 0
for i, v in enumerate(pca.explained_variance_ratio_):
    suma += v
    print(f"PC1..PC{i+1}: {suma:.4f} ({suma*100:.2f}%)")

# sprawdzamy ile ostatnich kolumn ma sume < 5%
print("\n=== ile kolumn usunac ===")
ratios = pca.explained_variance_ratio_
strata_1 = ratios[3]  # usuwamy 1 ostatnia
strata_2 = ratios[2] + ratios[3]  # usuwamy 2 ostatnie
print(f"strata po usunieciu 1 ostatniej (PC4): {strata_1*100:.2f}%")
print(f"strata po usunieciu 2 ostatnich (PC3+PC4): {strata_2*100:.2f}%")
print()
if strata_1 < 0.05:
    print("mozna usunac 1 kolumne (strata < 5%)")
if strata_2 < 0.05:
    print("mozna usunac 2 kolumny (strata < 5%)")

# transformujemy dane do 2 skladowych
pca2 = PCA(n_components=2)
X_2d = pca2.fit_transform(X)

# wykres 2d
kolory = {'Iris-setosa': 'blue', 'Iris-versicolor': 'orange', 'Iris-virginica': 'green'}
# sprawdzamy jak sa nazwane gatunki w pliku
unikalne = set(y)
print("\ngatunki w pliku:", unikalne)

fig, ax = plt.subplots()
for gatunek in set(y):
    maska = y == gatunek
    ax.scatter(X_2d[maska, 0], X_2d[maska, 1], label=gatunek, alpha=0.5, s=10)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA of IRIS dataset")
ax.legend()
plt.tight_layout()
plt.savefig("pca_2d.png")
plt.show()
print("zapisano pca_2d.png")

# wykres 3d
pca3 = PCA(n_components=3)
X_3d = pca3.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for gatunek in set(y):
    maska = y == gatunek
    ax.scatter(X_3d[maska, 0], X_3d[maska, 1], X_3d[maska, 2], label=gatunek, alpha=0.5, s=10)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA 3D of IRIS dataset")
ax.legend()
plt.tight_layout()
plt.savefig("pca_3d.png")
plt.show()
print("zapisano pca_3d.png")