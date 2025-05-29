import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

input_file = "income_data.txt"
names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target'
]

dataset = pd.read_csv(input_file, sep=',', header=None, names=names)

for column in dataset.select_dtypes(include=['float64', 'int64']).columns:
    dataset[column] = dataset[column].fillna(dataset[column].mean())

for column in dataset.select_dtypes(include=['object']).columns:
    dataset[column] = dataset[column].fillna(dataset[column].mode()[0])

label_encoders = {}
for column in dataset.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    dataset[column] = label_encoder.fit_transform(dataset[column])
    label_encoders[column] = label_encoder

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
scoring = 'accuracy'

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} - Accuracy: {accuracy:.4f}')
    
    y_test_inv = label_encoders['target'].inverse_transform(y_test)
    y_pred_inv = label_encoders['target'].inverse_transform(y_pred)
    print(classification_report(y_test_inv, y_pred_inv))
    
    results.append(accuracy)
    names.append(name)

plt.figure(figsize=(10, 6))
plt.bar(names, results)
plt.xlabel('Classification Algorithms')
plt.ylabel('Accuracy')
plt.title('Comparison of Classification Algorithms')
plt.show()

best_model_index = results.index(max(results))
print(f"Найкраща модель: {names[best_model_index]} з точністю {results[best_model_index]:.4f}")