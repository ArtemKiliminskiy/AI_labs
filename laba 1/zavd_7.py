import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.metrics import confusion_matrix

# Завантаження даних
iris = load_iris()
X, y = iris.data, iris.target

# Розділення даних
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Створення та навчання моделі
clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(X_train, y_train)

# Прогнозування
ypred = clf.predict(X_test)

# Виведення метрик
print('\nОцінки якості класифікації:')
print('-'*40)
print(f'Accuracy: {metrics.accuracy_score(y_test, ypred):.4f}')
print(f'Precision: {metrics.precision_score(y_test, ypred, average="weighted"):.4f}')
print(f'Recall: {metrics.recall_score(y_test, ypred, average="weighted"):.4f}')
print(f'F1 Score: {metrics.f1_score(y_test, ypred, average="weighted"):.4f}')
print(f'Cohen Kappa: {metrics.cohen_kappa_score(y_test, ypred):.4f}')
print(f'Matthews Corrcoef: {metrics.matthews_corrcoef(y_test, ypred):.4f}')

# Детальний звіт
print('\nДетальний звіт класифікації:')
print('-'*40)
print(metrics.classification_report(y_test, ypred))

# Матриця плутанини (без seaborn)
print('\nМатриця плутанини:')
mat = confusion_matrix(y_test, ypred)
print(mat)

# Візуалізація матриці за допомогою matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Матриця плутанини')
plt.colorbar()
tick_marks = np.arange(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names, rotation=45)
plt.yticks(tick_marks, iris.target_names)
plt.ylabel('Справжні значення')
plt.xlabel('Передбачені значення')

# Додавання значень у клітинки
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        plt.text(j, i, str(mat[i, j]),
                horizontalalignment="center",
                color="white" if mat[i, j] > mat.max()/2 else "black")

plt.tight_layout()

# Збереження результатів
plt.savefig("confusion_matrix.png")
with open("classification_results.txt", "w") as f:
    f.write("Результати класифікації:\n")
    f.write(f"Accuracy: {metrics.accuracy_score(y_test, ypred):.4f}\n")
    f.write(f"Precision: {metrics.precision_score(y_test, ypred, average='weighted'):.4f}\n")
    f.write(f"Recall: {metrics.recall_score(y_test, ypred, average='weighted'):.4f}\n")
    f.write(f"F1 Score: {metrics.f1_score(y_test, ypred, average='weighted'):.4f}\n")
    f.write("\nМатриця плутанини:\n")
    np.savetxt(f, mat, fmt='%d')

print('\nРезультати збережено у файлах: confusion_matrix.png та classification_results.txt')