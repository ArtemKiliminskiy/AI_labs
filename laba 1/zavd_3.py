import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# File handling and data loading
input_file = r'C:\laba 1\income_data.txt' 

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

try:
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
except FileNotFoundError:
    print(f"Error: File {input_file} not found!")
    exit(1)
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

X = np.array(X)

# Feature type identification
numeric_features = []
categorical_features = []

for i, item in enumerate(X[0]):
    if item.isdigit():
        numeric_features.append(i)
    else:
        categorical_features.append(i)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', preprocessing.StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Classifier pipeline
classifier_poly = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='poly', degree=8))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

# Model training and evaluation
classifier_poly.fit(X_train, y_train)
y_pred_poly = classifier_poly.predict(X_test)

# Results
print("Поліноміальне ядро:")
print(f"Акуратність: {accuracy_score(y_test, y_pred_poly) * 100:.2f}%")
print(f"Точність: {precision_score(y_test, y_pred_poly) * 100:.2f}%")
print(f"Повнота: {recall_score(y_test, y_pred_poly) * 100:.2f}%")
print(f"Результат F1: {f1_score(y_test, y_pred_poly) * 100:.2f}%")