import pandas as pd
import numpy as np

# a) sprawdzenie struktury plik ma blad w 3 wierszach
df = pd.read_csv("iris_big_with_errors.csv", on_bad_lines='skip')

print("a) struktura danych")
print("wiersze:", len(df))
print("kolumny:", df.columns.tolist())
print("oczekiwane wiersze: 1500, zaladowano:", len(df))

# b) brakujace dane i statystyki
print("\nb) brakujace dane")
print(df.isnull().sum())

# zamieniamy tekst na liczby a bledy staja sie NaN
num_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

print("\nstatystyki:")
print(df[num_cols].describe())

# c) dane spoza zakresu (0, 15) zastepujemy mediana kolumny
print("\nc) dane spoza zakresu")
for col in num_cols:
    poza = (df[col] < 0) | (df[col] > 15)
    print(f"{col}: {poza.sum()} bledow - wartosci: {df[col][poza].values}")
    # liczymy mediane tylko z poprawnych danych
    mediana = df[col][~poza & df[col].notna()].median()
    # zamieniamy bledy i NaN na mediane
    df[col] = df[col].where(~poza, mediana)
    df[col] = df[col].fillna(mediana)
    print(f"  -> zastapiono mediana: {mediana:.2f}")

# d) sprawdzenie gatunkow
print("\nd) gatunki")
print("znalezione wartosci:", df['target_name'].str.strip().unique())

naprawy = {
    'Setosa': 'setosa',
    'setosa.': 'setosa',
    'iris-setosa': 'setosa',
    'Versicolor': 'versicolor',
    'iris_versicolor': 'versicolor',
    'versicolr': 'versicolor',
    'versi-color': 'versicolor',
    'VIRGINICA': 'virginica',
    'iris virginica': 'virginica',
    'virginica?': 'virginica',
    'unknown': np.nan  # nie wiadomo co to zostawiamy NaN
}

df['target_name'] = df['target_name'].str.strip()
df['target_name'] = df['target_name'].replace(naprawy)

print("\npo naprawie:", df['target_name'].unique())
print("brakujace target:", df['target_name'].isnull().sum())

# usuwamy wiersze gdzie nie wiemy jaki gatunek
df = df.dropna(subset=['target_name'])
print("wierszy po usunieciu unknown:", len(df))

# zapis do pliku
df.to_csv("iris_fixed.csv", index=False)
print("\nzapisano do iris_fixed.csv")
print(df.describe())