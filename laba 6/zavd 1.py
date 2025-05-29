import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Крок 1: Створення набору даних
# Дані з таблиці у завданні
data = {
    'День': ['Сонячно', 'Сонячно', 'Хмарно', 'Дощ', 'Дощ', 'Дощ', 'Хмарно', 'Сонячно', 'Сонячно', 'Дощ', 'Сонячно', 'Хмарно', 'Хмарно', 'Дощ'],
    'Температура': ['Висока', 'Висока', 'Висока', 'Середня', 'Низька', 'Низька', 'Низька', 'Середня', 'Низька', 'Середня', 'Середня', 'Середня', 'Висока', 'Середня'],
    'Вологість': ['Висока', 'Висока', 'Висока', 'Висока', 'Нормальна', 'Нормальна', 'Нормальна', 'Висока', 'Нормальна', 'Нормальна', 'Нормальна', 'Висока', 'Нормальна', 'Висока'],
    'Вітер': ['Ні', 'Так', 'Ні', 'Ні', 'Ні', 'Так', 'Так', 'Ні', 'Ні', 'Ні', 'Так', 'Так', 'Ні', 'Так'],
    'Гра': ['Ні', 'Ні', 'Так', 'Так', 'Так', 'Ні', 'Так', 'Ні', 'Так', 'Так', 'Так', 'Так', 'Так', 'Ні']
}

df = pd.DataFrame(data)
print("Набір даних:")
print(df)
print("\n")

# Крок 2: Створення частотних таблиць та таблиць правдоподібності
def create_frequency_and_likelihood_tables(df, column):
    # Частотна таблиця
    frequency_table = pd.crosstab(df[column], df['Гра'])
    
    # Додаємо суму по рядках
    frequency_table['Всього'] = frequency_table.sum(axis=1)
    
    # Таблиця правдоподібності (ймовірності)
    likelihood_table = frequency_table.copy()
    total_yes = frequency_table['Так'].sum()
    total_no = frequency_table['Ні'].sum()
    total = total_yes + total_no
    
    likelihood_table['Так'] = likelihood_table['Так'] / total_yes
    likelihood_table['Ні'] = likelihood_table['Ні'] / total_no
    likelihood_table['Всього'] = likelihood_table['Всього'] / total
    
    return frequency_table, likelihood_table

# Обчислення для кожного атрибуту
print("Частотні таблиці та таблиці правдоподібності:")
for column in ['День', 'Температура', 'Вологість', 'Вітер']:
    print(f"\nДля атрибуту '{column}':")
    freq_table, like_table = create_frequency_and_likelihood_tables(df, column)
    print("\nЧастотна таблиця:")
    print(freq_table)
    print("\nТаблиця правдоподібності:")
    print(like_table)

# Крок 3: Загальні ймовірності (Prior)
total_records = len(df)
p_yes = df[df['Гра'] == 'Так'].shape[0] / total_records
p_no = df[df['Гра'] == 'Ні'].shape[0] / total_records

print("\nЗагальні ймовірності:")
print(f"P(Гра=Так) = {p_yes:.2f}")
print(f"P(Гра=Ні) = {p_no:.2f}")

# Крок 4: Використання наївного Байєса з scikit-learn
print("\nРеалізація за допомогою scikit-learn:")

# Підготовка даних
features = df.drop('Гра', axis=1)
target = df['Гра']

# Конвертація категоріальних ознак в числові
encoders = {}
for column in features.columns:
    encoders[column] = LabelEncoder()
    features[column] = encoders[column].fit_transform(features[column])

# Конвертація цільової змінної
target_encoder = LabelEncoder()
target_encoded = target_encoder.fit_transform(target)

# Розділення даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.3, random_state=42)

# Навчання моделі
nb_classifier = CategoricalNB()
nb_classifier.fit(X_train, y_train)

# Прогнозування на тестовому наборі
y_pred = nb_classifier.predict(X_test)

# Оцінка моделі
print(f"Точність моделі: {accuracy_score(y_test, y_pred):.2f}")
print("\nЗвіт про класифікацію:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# Крок 5: Ручний розрахунок ймовірностей для нового прикладу
print("\nРучний розрахунок для нового прикладу:")
new_example = {
    'День': 'Хмарно',
    'Температура': 'Середня',
    'Вологість': 'Висока',
    'Вітер': 'Так'
}
print(f"Новий приклад: {new_example}")

# Розрахунок P(X|Гра=Так) * P(Гра=Так)
def calculate_probability(example, class_value):
    # Ймовірність класу
    class_prob = p_yes if class_value == 'Так' else p_no
    
    # Умовні ймовірності для кожного атрибуту
    conditional_probs = []
    for attr, value in example.items():
        # Отримуємо таблицю правдоподібності для атрибуту
        _, like_table = create_frequency_and_likelihood_tables(df, attr)
        
        # Якщо значення відсутнє в таблиці, використовуємо лапласове згладжування
        try:
            conditional_prob = like_table.loc[value, class_value]
        except KeyError:
            # Виконуємо лапласове згладжування
            unique_values = df[attr].nunique()
            if class_value == 'Так':
                conditional_prob = 1 / (df[df['Гра'] == 'Так'].shape[0] + unique_values)
            else:
                conditional_prob = 1 / (df[df['Гра'] == 'Ні'].shape[0] + unique_values)
                
        conditional_probs.append(conditional_prob)
    
    # Розрахунок P(X|Гра=клас) * P(Гра=клас)
    return class_prob * np.prod(conditional_probs)

# Обчислюємо P(X|Гра=Так) * P(Гра=Так)
p_yes_given_x = calculate_probability(new_example, 'Так')
print(f"P(X|Гра=Так) * P(Гра=Так) = {p_yes_given_x:.6f}")

# Обчислюємо P(X|Гра=Ні) * P(Гра=Ні)
p_no_given_x = calculate_probability(new_example, 'Ні')
print(f"P(X|Гра=Ні) * P(Гра=Ні) = {p_no_given_x:.6f}")

# Нормалізуємо ймовірності
p_yes_normalized = p_yes_given_x / (p_yes_given_x + p_no_given_x)
p_no_normalized = p_no_given_x / (p_yes_given_x + p_no_given_x)

print("\nНормалізовані ймовірності:")
print(f"P(Гра=Так|X) = {p_yes_normalized:.4f} або {p_yes_normalized*100:.2f}%")
print(f"P(Гра=Ні|X) = {p_no_normalized:.4f} або {p_no_normalized*100:.2f}%")

print("\nПрогноз: Гра {'відбудеться' if p_yes_normalized > p_no_normalized else 'не відбудеться'}")