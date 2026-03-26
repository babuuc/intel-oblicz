import os
import pandas as pd

def normalize_species(value):
    text = str(value)
    text = text.strip()
    text = text.lower()

    # usuwamy najczestsze znaki specjalne i prefiksy
    text = text.replace("iris", "")
    text = text.replace("_", "")
    text = text.replace("-", "")
    text = text.replace(" ", "")
    text = text.replace(".", "")
    text = text.replace("?", "")

    if ("setosa" in text) == True:
        return "setosa"
    if ("versicolor" in text) == True or ("versicolr" in text) == True:
        return "versicolor"
    if ("virginica" in text) == True:
        return "virginica"
    return None

base_dir = os.path.dirname(os.path.abspath(__file__))
input_path = base_dir + "/iris_big_with_errors.csv"
output_path = base_dir + "/iris_big_no_errors.csv"

# skipujemy bledne linie
df = pd.read_csv(input_path, on_bad_lines="skip")

numeric_columns = []
numeric_columns.append(df.columns[0])
numeric_columns.append(df.columns[1])
numeric_columns.append(df.columns[2])
numeric_columns.append(df.columns[3])
target_column = df.columns[4]

# a
print("[a] liczba pustych wartosci przed naprawa:")
print(df.isna().sum())

for i in range(len(numeric_columns)):
    col = numeric_columns[i]
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

# b
for i in range(len(numeric_columns)):
    col = numeric_columns[i]
    median = df[col].median()
    df[col] = df[col].fillna(median)

    for index in df.index:
        if df[col][index] <= 0 or df[col][index] >= 15:
            df.loc[index, col] = median

# c
df[target_column] = df[target_column].apply(normalize_species)

before_drop = len(df)
df = df[df[target_column].notna()]
removed_rows = before_drop - len(df)

# d
df.to_csv(output_path, index=False)

print("[d] usuniete wiersze z niepoprawna klasa: " + str(removed_rows))
print("[d] zapisano oczyszczony plik: " + output_path.split("/")[-1])
print("liczba rekordow po czyszczeniu: " + str(len(df)))