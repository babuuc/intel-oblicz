import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv("iris_big.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=13)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

labels = ["setosa", "versicolor", "virginica"]

classifiers = [
    ("drzewo decyzyjne", DecisionTreeClassifier()),
    ("knn k=3", KNeighborsClassifier(n_neighbors=3)),
    ("knn k=5", KNeighborsClassifier(n_neighbors=5)),
    ("knn k=11", KNeighborsClassifier(n_neighbors=11)),
    ("naive bayes", GaussianNB()),
    ("mlp", MLPClassifier(max_iter=1000)),
]

for name, clf in classifiers:
    clf.fit(train_inputs, train_classes)
    acc = clf.score(test_inputs, test_classes)
    preds = clf.predict(test_inputs)
    cm = confusion_matrix(test_classes, preds, labels=labels)
    print("klasyfikator:", name)
    print("dokladnosc:", round(acc * 100, 2), "%")
    print("macierz bledow:")
    print(cm)
    print()