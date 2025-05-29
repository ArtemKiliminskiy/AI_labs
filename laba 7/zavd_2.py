import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode

# Завантаження даних Iris
iris = load_iris()
X = iris.data
y = iris.target

# Стандартизація даних
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Кластеризація методом k-середніх
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Візуалізація результатів
plt.figure(figsize=(12, 5))

# Графік для перших двох ознак
plt.subplot(121)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('Кластеризація Iris (ознаки 1-2)')
plt.xlabel('Стандартизована довжина чашолистка')
plt.ylabel('Стандартизована ширина чашолистка')

# Графік для інших двох ознак
plt.subplot(122)
plt.scatter(X_scaled[:, 2], X_scaled[:, 3], c=kmeans.labels_, cmap='viridis')
plt.title('Кластеризація Iris (ознаки 3-4)')
plt.xlabel('Стандартизована довжина пелюстки')
plt.ylabel('Стандартизована ширина пелюстки')

plt.tight_layout()
plt.show()

# Оцінка точності кластеризації
from sklearn.metrics import accuracy_score

def get_cluster_labels(kmeans_labels, true_labels):
    cluster_labels = np.zeros_like(kmeans_labels)
    for i in range(3):
        mask = (kmeans_labels == i)
        cluster_labels[mask] = mode(true_labels[mask], keepdims=False)[0]
    return cluster_labels

cluster_labels = get_cluster_labels(kmeans.labels_, y)
print("Точність кластеризації:", accuracy_score(y, cluster_labels))