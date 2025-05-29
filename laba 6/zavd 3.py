import pandas as pd
import numpy as np

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

# Враховуючи, що в даних "Ні" - відповідає "Слабкий вітер", а "Так" - "Сильний вітер",
# Для кращого розуміння переведемо в українські значення:
translate_dict = {
    'День': {'Сонячно': 'Сонячно', 'Хмарно': 'Похмуро', 'Дощ': 'Дощ'},
    'Вітер': {'Ні': 'Слабкий', 'Так': 'Сильний'}
}

for col, mapping in translate_dict.items():
    df[col] = df[col].map(lambda x: mapping.get(x, x))

print("Набір даних з українськими значеннями:")
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

# Крок 4: Розв'язання 
# День (Outlook) = Sunny (Сонячно)
# Вологість (Humidity) = High (Висока)
# Вітер (Wind) = Weak (Слабкий)

new_example = {
    'День': 'Сонячно',
    'Вологість': 'Висока',
    'Вітер': 'Слабкий'
}

print("\nРозрахунок ймовірностей :")
print(f"Умови: {new_example}")

# Отримання потрібних ймовірностей з таблиць правдоподібності
# Ймовірність P(День=Сонячно|Гра=Так)
_, day_like_table = create_frequency_and_likelihood_tables(df, 'День')
p_day_given_yes = day_like_table.loc['Сонячно', 'Так']
p_day_given_no = day_like_table.loc['Сонячно', 'Ні']

# Ймовірність P(Вологість=Висока|Гра=Так)
_, humidity_like_table = create_frequency_and_likelihood_tables(df, 'Вологість')
p_humidity_given_yes = humidity_like_table.loc['Висока', 'Так']
p_humidity_given_no = humidity_like_table.loc['Висока', 'Ні']

# Ймовірність P(Вітер=Слабкий|Гра=Так)
_, wind_like_table = create_frequency_and_likelihood_tables(df, 'Вітер')
p_wind_given_yes = wind_like_table.loc['Слабкий', 'Так']
p_wind_given_no = wind_like_table.loc['Слабкий', 'Ні']

# Розрахунок P(X|Гра=Так) * P(Гра=Так)
p_yes_x = p_day_given_yes * p_humidity_given_yes * p_wind_given_yes * p_yes
print(f"P(X|Гра=Так) * P(Гра=Так) = {p_day_given_yes:.4f} * {p_humidity_given_yes:.4f} * {p_wind_given_yes:.4f} * {p_yes:.4f} = {p_yes_x:.6f}")

# Розрахунок P(X|Гра=Ні) * P(Гра=Ні)
p_no_x = p_day_given_no * p_humidity_given_no * p_wind_given_no * p_no
print(f"P(X|Гра=Ні) * P(Гра=Ні) = {p_day_given_no:.4f} * {p_humidity_given_no:.4f} * {p_wind_given_no:.4f} * {p_no:.4f} = {p_no_x:.6f}")

# Обчислення повної ймовірності P(X)
p_x = p_yes_x + p_no_x

# Нормалізовані ймовірності P(Гра=Так|X) і P(Гра=Ні|X)
p_yes_given_x = p_yes_x / p_x
p_no_given_x = p_no_x / p_x

print("\nНормалізовані ймовірності:")
print(f"P(Гра=Так|X) = {p_yes_given_x:.4f} або {p_yes_given_x*100:.2f}%")
print(f"P(Гра=Ні|X) = {p_no_given_x:.4f} або {p_no_given_x*100:.2f}%")

# Висновок
if p_yes_given_x > p_no_given_x:
    conclusion = "Так, матч відбудеться"
else:
    conclusion = "Ні, матч не відбудеться"

print(f"\nВисновок : {conclusion}")