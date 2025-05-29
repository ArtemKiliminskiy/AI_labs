import numpy as np
from sklearn import preprocessing, metrics
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split

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
label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i].astype(int)
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])
        label_encoder.append(encoder)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=5
)

classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

f1 = metrics.f1_score(y_test, y_pred, average='weighted')
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')

print(f"Результат F1: {f1:.2%}")
print(f"Акуратність: {accuracy:.2%}")
print(f"Точність: {precision:.2%}")
print(f"Повнота: {recall:.2%}")

input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

input_data_encoded = []
encoder_index = 0

for i, item in enumerate(input_data):
    if X[0, i].isdigit():
        input_data_encoded.append(int(item))
    else:
        try:
            encoded_value = label_encoder[encoder_index].transform([item])[0]
        except ValueError:
            raise ValueError(f"Нове значення '{item}' відсутнє в кодувальнику для ознаки {i}.")
        input_data_encoded.append(encoded_value)
        encoder_index += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)
predicted_class = classifier.predict(input_data_encoded)

result = '>50K' if predicted_class[0] == 1 else '<=50K'
print(f"Результат класифікації: {result}")