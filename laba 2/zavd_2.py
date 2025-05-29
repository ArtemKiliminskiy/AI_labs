import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

input_file = 'income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append(0)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(1)
            count_class2 += 1

X = np.array(X)
numeric_features = []
categorical_features = []

for i, item in enumerate(X[0]):
    if item.isdigit():
        numeric_features.append(i)
    else:
        categorical_features.append(i)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', preprocessing.StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

classifier_poly = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='poly', degree=8))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

classifier_poly.fit(X_train, y_train)
y_pred_poly = classifier_poly.predict(X_test)

print("Поліноміальне ядро:")
print(f"Акуратність: {accuracy_score(y_test, y_pred_poly) * 100:.2f}%")
print(f"Точність: {precision_score(y_test, y_pred_poly) * 100:.2f}%")
print(f"Повнота: {recall_score(y_test, y_pred_poly) * 100:.2f}%")
print(f"Результат F1: {f1_score(y_test, y_pred_poly) * 100:.2f}%")