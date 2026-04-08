import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix

df = pd.read_csv("iris_big.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=300832)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

tree = DecisionTreeClassifier()
tree.fit(train_inputs, train_classes)

print("drzewo decyzyjne:")
print(export_text(tree, feature_names=["sl", "sw", "pl", "pw"]))

acc = tree.score(test_inputs, test_classes)
print("dokladnosc:", round(acc * 100, 2), "%")

predictions = tree.predict(test_inputs)
cm = confusion_matrix(test_classes, predictions, labels=["setosa", "versicolor", "virginica"])
print("macierz bledow (setosa, versicolor, virginica):")
print(cm)