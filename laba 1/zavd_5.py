from pandas import read_csv
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Завантаження даних
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Експлораторний аналіз даних
print("Перші 20 записів:")
print(dataset.head(20))
print("\nСтатистичний опис:")
print(dataset.describe())
print("\nРозподіл за класами:")
print(dataset.groupby('class').size())

# Візуалізація даних
dataset.plot(kind='box', subplots=True, layout=(2, 2), 
             sharex=False, sharey=False)
plt.show()

dataset.hist()
plt.show()

scatter_matrix(dataset)
plt.show()

# Підготовка даних
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1
)

# Налаштування крос-валідації
kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

# Ініціалізація моделей
models = [
    ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

# Оцінка моделей
results = []
names = []
print("\nРезультати крос-валідації:")
for name, model in models:
    cv_results = cross_val_score(
        model, X_train, Y_train, cv=kfold, scoring='accuracy'
    )
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} (±{cv_results.std():.4f})")

# Вибір найкращої моделі
best_index = results.index(max(results))
best_name = names[best_index]
best_model = models[best_index][1]
best_model.fit(X_train, Y_train)

# Оцінка на тестовому наборі
predictions = best_model.predict(X_validation)
print("\nНайкраща модель:", best_name)
print("Точність:", accuracy_score(Y_validation, predictions))
print("Матриця помилок:\n", confusion_matrix(Y_validation, predictions))
print("Звіт класифікації:\n", 
      classification_report(Y_validation, predictions))