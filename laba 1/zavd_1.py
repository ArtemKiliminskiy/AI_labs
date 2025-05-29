import numpy as np
from sklearn import preprocessing, metrics
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
import os

# Можливі розташування файлу
base_paths = [
    "C:\\loxi ebani",
    "C:\\loxi_ebani",
    "C:\\Users\\artem\\loxi ebani",
    "C:\\Users\\artem\\loxi_ebani",
    "C:\\"
]

# Назва файлу
filename = "income_data.txt"
input_file = None

# Спочатку перевіряємо конкретні шляхи
for path in base_paths:
    potential_file = os.path.join(path, filename)
    if os.path.exists(potential_file):
        input_file = potential_file
        print(f"Знайдено файл: {input_file}")
        break

# Якщо файл не знайдено у вказаних папках, шукаємо в корені диску C
if input_file is None:
    print("Файл не знайдено за вказаними шляхами. Починаємо пошук на диску C...")
    input_file = find_file(filename, "C:\\")
    if input_file:
        print(f"Знайдено файл: {input_file}")
    else:
        print("Файл не знайдено автоматично. Будь ласка, вкажіть повний шлях до файлу.")
        input_file = input("Введіть повний шлях до файлу income_data.txt: ")

# Перевірка наявності файлу
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Файл не знайдено: {input_file}")

print(f"Використовуємо файл: {input_file}")

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

# Спробуємо декілька варіантів кодування для відкриття файлу
encodings = ['utf-8', 'cp1251', 'latin1']
file_opened = False

for encoding in encodings:
    try:
        with open(input_file, 'r', encoding=encoding) as f:
            print(f"Файл успішно відкрито з кодуванням {encoding}")
            for line in f.readlines():
                if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
                    break
                if '?' in line:
                    continue
                
                data = line.strip().split(', ')
                if len(data) < 2:  # Перевірка чи рядок містить дані
                    continue
                    
                if data[-1] == '<=50K' and count_class1 < max_datapoints:
                    X.append(data[:-1])
                    y.append(0)
                    count_class1 += 1
                elif data[-1] == '>50K' and count_class2 < max_datapoints:
                    X.append(data[:-1])
                    y.append(1)
                    count_class2 += 1
            
            file_opened = True
            break
    except UnicodeDecodeError:
        print(f"Не вдалося відкрити файл з кодуванням {encoding}")
    except Exception as e:
        print(f"Інша помилка при відкритті файлу: {e}")

if not file_opened:
    raise ValueError("Не вдалося відкрити файл з жодним з відомих кодувань")

# Перевірка, чи є дані
if not X or not y:
    raise ValueError("Не вдалося завантажити дані з файлу. Перевірте шлях та формат файлу.")

X = np.array(X)
y = np.array(y)

# Перевірка розмірності даних
print(f"Завантажено записів: {len(X)}")
print(f"Розподіл класів: {count_class1} записів <=50K, {count_class2} записів >50K")

# Створюємо масив для кодованих даних
X_encoded = np.empty(X.shape, dtype=object)
label_encoder = []

# Кодуємо категоріальні змінні
for i in range(X.shape[1]):
    # Перевіряємо тип ознаки (числова чи категоріальна)
    try:
        X_encoded[:, i] = X[:, i].astype(float)
    except ValueError:
        # Якщо не можемо конвертувати в число, значить це категоріальна ознака
        current_encoder = preprocessing.LabelEncoder()
        current_encoder.fit(X[:, i])
        X_encoded[:, i] = current_encoder.transform(X[:, i])
        label_encoder.append((i, current_encoder))

# Конвертуємо у float для коректної роботи з алгоритмами
X_encoded = X_encoded.astype(float)

# Розділяємо дані на тренувальні та тестові
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=5
)

# Навчаємо класифікатор
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=5000))
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Оцінюємо результати
f1 = metrics.f1_score(y_test, y_pred, average='weighted')
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')

print(f"Результат F1: {f1:.2%}")
print(f"Акуратність: {accuracy:.2%}")
print(f"Точність: {precision:.2%}")
print(f"Повнота: {recall:.2%}")

# Класифікація нового прикладу
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
              '0', '0', '40', 'United-States']

input_data_encoded = np.empty(len(input_data), dtype=float)

# Кодуємо вхідні дані так само, як і навчальні
for i in range(len(input_data)):
    try:
        input_data_encoded[i] = float(input_data[i])
    except ValueError:
        # Знаходимо відповідний кодувальник для цієї ознаки
        for idx, encoder in label_encoder:
            if idx == i:
                try:
                    input_data_encoded[i] = encoder.transform([input_data[i]])[0]
                except ValueError:
                    # Якщо значення немає в кодувальнику, використовуємо -1
                    print(f"Увага: Значення '{input_data[i]}' не знайдено для ознаки {i}. Встановлено -1.")
                    input_data_encoded[i] = -1
                break

# Класифікуємо
input_data_encoded = input_data_encoded.reshape(1, -1)
predicted_class = classifier.predict(input_data_encoded)
result = '>50K' if predicted_class[0] == 1 else '<=50K'

print(f"Результат класифікації: {result}")

print("\nПрограма успішно завершена!")