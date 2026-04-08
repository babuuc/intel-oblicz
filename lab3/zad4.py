import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df = pd.read_csv("diagnosis.csv")

# a) wykres 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = df['diagnosis'].map({0: 'blue', 1: 'red'})
ax.scatter(df['param1'], df['param2'], df['param3'], c=colors, alpha=0.5)
ax.set_xlabel('param1')
ax.set_ylabel('param2')
ax.set_zlabel('param3')
plt.title('dane diagnostyczne (niebieski=zdrowy, czerwony=chory)')
plt.savefig('wykres3d.png')
plt.close()
print("zapisano wykres3d.png")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=13)

train_inputs = train_set[:, 0:3].astype(float)
train_classes = train_set[:, 3].astype(int)
test_inputs = test_set[:, 0:3].astype(float)
test_classes = test_set[:, 3].astype(int)

classifiers = [
    ("drzewo decyzyjne", DecisionTreeClassifier()),
    ("knn k=3", KNeighborsClassifier(n_neighbors=3)),
    ("knn k=5", KNeighborsClassifier(n_neighbors=5)),
    ("knn k=11", KNeighborsClassifier(n_neighbors=11)),
    ("naive bayes", GaussianNB()),
    ("mlp", MLPClassifier(max_iter=1000)),
]

# b) dla kazdego klasyfikatora: accuracy, precision, recall + confusion matrix
for name, clf in classifiers:
    clf.fit(train_inputs, train_classes)
    preds = clf.predict(test_inputs)

    acc = accuracy_score(test_classes, preds)
    prec = precision_score(test_classes, preds)
    rec = recall_score(test_classes, preds)
    cm = confusion_matrix(test_classes, preds)

    print("klasyfikator:", name)
    print("accuracy:", round(acc * 100, 2), "%")
    print("precision:", round(prec * 100, 2), "%")
    print("recall:", round(rec * 100, 2), "%")
    print("macierz bledow:")
    print(cm)

    # seaborn heatmap
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['zdrowy', 'chory'],
                yticklabels=['zdrowy', 'chory'])
    plt.title(name)
    plt.ylabel('prawdziwa klasa')
    plt.xlabel('przewidywana klasa')
    plt.tight_layout()
    filename = 'cm_' + name.replace(' ', '_').replace('=', '') + '.png'
    plt.savefig(filename)
    plt.close()
    print("zapisano", filename)
    print()

# c) odpowiedzi na pytania

# accuracy = (tp+tn)/(wszystkie) - ogolna dokladnosc
# precision = tp/(tp+fp) - z tych co przewidzielismy jako chorych ile faktycznie chorych
# recall = tp/(tp+fn) - z tych co sa faktycznie chorzy ile znalezlismy
# precision wazna gdy nie chcemy klasyfikowac zdrowych jako chorych
# recall wazna gdy chcemy minimalizowac ryzyko ze chorego powiemy ze zdrowy
#
# d) niezbalansowany zbior: accuracy moze byc mylaaca
# np 900 zdrowych, 100 chorych - klasyfikator zwracajacy zawsze 0 ma 90% accuracy
# ale recall = 0% - nie wykrywa zadnego chorego