import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

input_file = r'C:\laba 1\income_data.txt' 

# Initialize variables
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

# Load and process data
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

# Identify feature types
numeric_features = []
categorical_features = []

for i, item in enumerate(X[0]):
    try:
        float(item)
        numeric_features.append(i)
    except ValueError:
        categorical_features.append(i)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', preprocessing.StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create classifier with sigmoid kernel
classifier_sigmoid = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='sigmoid'))
])

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

classifier_sigmoid.fit(X_train, y_train)
y_pred_sigmoid = classifier_sigmoid.predict(X_test)

# Print results
print("\nСигмоїдальне ядро:")
print(f"Акуратність: {accuracy_score(y_test, y_pred_sigmoid) * 100:.2f}%")
print(f"Точність: {precision_score(y_test, y_pred_sigmoid, average='weighted') * 100:.2f}%")
print(f"Повнота: {recall_score(y_test, y_pred_sigmoid, average='weighted') * 100:.2f}%")
print(f"Результат F1: {f1_score(y_test, y_pred_sigmoid, average='weighted') * 100:.2f}%")