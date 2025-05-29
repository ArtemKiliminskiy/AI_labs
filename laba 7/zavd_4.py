import datetime
import json
import numpy as np
import yfinance as yf
from sklearn import covariance, cluster
import warnings
warnings.filterwarnings('ignore')

# Input file containing company symbols
input_file = 'company_symbol_mapping.json'

# Load the company symbol map
try:
    with open(input_file, 'r') as f:
        company_symbols_map = json.loads(f.read())
    print(f"Завантажено {len(company_symbols_map)} символів компаній")
except FileNotFoundError:
    print(f"Файл {input_file} не знайдено!")
    exit()

symbols, names = np.array(list(company_symbols_map.items())).T

# Load the historical stock quotes
start_date = datetime.datetime(2003, 7, 3)
end_date = datetime.datetime(2007, 5, 4)

print(f"Завантаження даних з {start_date.date()} по {end_date.date()}...")

# Отримання даних через yfinance
quotes = []
successful_symbols = []
successful_names = []

for i, symbol in enumerate(symbols):
    try:
        print(f"Завантаження {symbol} ({i+1}/{len(symbols)})...")
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if not data.empty and len(data) > 10:
            # Перевіряємо, чи є колонки Open та Close
            if 'Open' in data.columns and 'Close' in data.columns:
                quotes.append(data)
                successful_symbols.append(symbol)
                successful_names.append(names[i])
            else:
                print(f"Відсутні колонки Open/Close для {symbol}")
        else:
            print(f"Недостатньо даних для {symbol}")
            
    except Exception as e:
        print(f"Помилка завантаження даних для {symbol}: {e}")

print(f"\nУспішно завантажено дані для {len(quotes)} компаній з {len(symbols)}")

if len(quotes) < 2:
    print("Недостатньо даних для кластеризації!")
    exit()

# Знайти спільний діапазон дат для всіх акцій
common_dates = quotes[0].index
for quote in quotes[1:]:
    common_dates = common_dates.intersection(quote.index)

print(f"Спільних дат: {len(common_dates)}")

if len(common_dates) < 10:
    print("Недостатньо спільних дат для аналізу!")
    exit()

# Вилучити дані за спільними датами
aligned_quotes = []
for quote in quotes:
    aligned_data = quote.loc[common_dates]
    aligned_quotes.append(aligned_data)

# Extract opening and closing quotes
try:
    opening_quotes = np.array([quote['Open'].values for quote in aligned_quotes]).astype(np.float64)
    closing_quotes = np.array([quote['Close'].values for quote in aligned_quotes]).astype(np.float64)
    
    print(f"Розмір матриці даних: {opening_quotes.shape}")
    
except Exception as e:
    print(f"Помилка при обробці цін: {e}")
    exit()

# Compute differences between opening and closing quotes
quotes_diff = closing_quotes - opening_quotes

# Перевірити на наявність NaN значень
print(f"Кількість NaN в opening_quotes: {np.isnan(opening_quotes).sum()}")
print(f"Кількість NaN в closing_quotes: {np.isnan(closing_quotes).sum()}")
print(f"Кількість NaN в quotes_diff: {np.isnan(quotes_diff).sum()}")

# Видалити компанії, які мають багато NaN значень
nan_threshold = 0.1  # Якщо більше 10% даних - NaN, видаляємо компанію
nan_ratio_per_company = np.isnan(quotes_diff).mean(axis=0)
valid_companies = nan_ratio_per_company < nan_threshold

quotes_diff = quotes_diff[:, valid_companies]
successful_names = np.array(successful_names)[valid_companies]

print(f"Після фільтрації компаній: {quotes_diff.shape}")
print(f"Залишилось компаній: {len(successful_names)}")

if quotes_diff.shape[1] < 2:
    print("Недостатньо компаній після фільтрації!")
    exit()

# Заповнити оставшиеся NaN значення
for i in range(quotes_diff.shape[1]):
    company_data = quotes_diff[:, i]
    if np.isnan(company_data).any():
        # Заповнюємо NaN медіанним значенням
        median_val = np.nanmedian(company_data)
        quotes_diff[:, i] = np.where(np.isnan(company_data), median_val, company_data)

print(f"Після заповнення NaN: {np.isnan(quotes_diff).sum()} NaN значень")

# Normalize the data
X = quotes_diff.copy().T
X_std = X.std(axis=0)
X_std[X_std == 0] = 1  # Уникнення ділення на нуль
X /= X_std

print(f"Нормалізовані дані: {X.shape}")

# Create a graph model
try:
    # Зменшуємо складність для невеликих наборів даних
    if X.shape[0] < 10:
        edge_model = covariance.EmpiricalCovariance()
    else:
        edge_model = covariance.GraphLassoCV(cv=min(3, X.shape[0]//2))
    
    # Train the model
    with np.errstate(invalid='ignore'):
        edge_model.fit(X)
    
    print("Модель коваріації успішно навчена")
    
except Exception as e:
    print(f"Помилка при навчанні моделі коваріації: {e}")
    # Використовуємо найпростіший підхід
    print("Використовуємо кореляційну матрицю...")
    covariance_matrix = np.corrcoef(X)
    edge_model = type('obj', (object,), {'covariance_': covariance_matrix})

# Build clustering model using Affinity Propagation
try:
    print("Виконання кластеризації...")
    _, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=42)
    num_labels = labels.max()
    
    print(f"Знайдено {num_labels + 1} кластерів")
    
except Exception as e:
    print(f"Помилка при кластеризації Affinity Propagation: {e}")
    # Альтернативний метод кластеризації
    print("Використовуємо KMeans як альтернативу...")
    from sklearn.cluster import KMeans
    n_clusters = min(5, len(successful_names))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    num_labels = labels.max()

# Print the results of clustering
print('\n' + '='*60)
print('РЕЗУЛЬТАТИ КЛАСТЕРИЗАЦІЇ АКЦІЙ')
print('На основі різниці між цінами відкриття та закриття')
print('='*60)

for i in range(num_labels + 1):
    cluster_companies = successful_names[labels == i]
    print(f"\nКластер {i+1} ({len(cluster_companies)} компаній):")
    print("  " + ', '.join(cluster_companies))

# Додаткова статистика
print(f'\nЗагальна статистика:')
print(f'- Загальна кількість компаній: {len(successful_names)}')
print(f'- Кількість кластерів: {num_labels + 1}')
print(f'- Період аналізу: {start_date.date()} - {end_date.date()}')
print(f'- Кількість торгових днів: {quotes_diff.shape[0]}')

# Показати розмір кожного кластера
cluster_sizes = [np.sum(labels == i) for i in range(num_labels + 1)]
print(f'- Розміри кластерів: {cluster_sizes}')

# Показати статистику по кластерах
print(f'\nСередня різниця Open-Close по кластерах:')
for i in range(num_labels + 1):
    cluster_mask = labels == i
    cluster_data = quotes_diff[:, cluster_mask]
    mean_diff = np.mean(cluster_data)
    std_diff = np.std(cluster_data)
    print(f'Кластер {i+1}: середня = {mean_diff:.4f}, стд. відхилення = {std_diff:.4f}')