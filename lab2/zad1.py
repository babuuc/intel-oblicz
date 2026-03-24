import pandas as pd
import re

pattern = re.compile(r'^\d\.\d{2},"\d\.\d{2}","\d\.\d{2}","\d\.\d{2}","(?:setosa|versicolor|virginica)"$')

valid_lines = []

with open('iris_big_with_errors.csv', 'r', encoding='utf-8') as f:
    for line in f:
        clean_line = line.strip()
        if pattern.match(clean_line):
            valid_lines.append(clean_line)

with open('iris_big_no_errors.csv', 'w', encoding='utf-8') as f:
    f.write('\n'.join(valid_lines) + '\n')

df_clean = pd.read_csv(
    'iris_big_no_errors.csv', 
    header=None, 
    names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
)

print("Czyszczenie zakończone. Oto statystyki oczyszczonej bazy danych:")
print(df_clean.info())
