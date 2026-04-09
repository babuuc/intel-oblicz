import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris_big.csv")
print("wczytano dane")
print(df)

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=300832)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

print("zbior treningowy:", train_set.shape[0], "rekordow")
print("zbior testowy:", test_set.shape[0], "rekordow")

# setosa ma krotkie platki (petal length < 2)
# versicolor ma waskie platki (petal width < 1.5)
# virginica to po prost reszta

def classify_iris(sl, sw, pl, pw):
    if float(pl) < 2:
        return "setosa"
    elif float(pw) < 1.5:
        return "versicolor"
    else:
        return "virginica"

good_predictions = 0
n = test_set.shape[0]
for i in range(n):
    if classify_iris(test_set[i, 0], test_set[i, 1], test_set[i, 2], test_set[i, 3]) == test_set[i, 4]:
        good_predictions = good_predictions + 1

print("poprawne:", good_predictions)
print("dokladnosc:", round(good_predictions / n * 100, 2), "%")