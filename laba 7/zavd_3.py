import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# Завантаження вхідних даних
data = np.loadtxt('data_clustering.txt', delimiter=',')

# Оцінка ширини вікна
bandwidth = estimate_bandwidth(data, quantile=0.2)

# Кластеризація методом зсуву середнього
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
mean_shift.fit(data)

# Отримання центрів кластерів та міток
cluster_centers = mean_shift.cluster_centers_
labels = mean_shift.labels_
n_clusters = len(cluster_centers)

# Візуалізація результатів
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            c='red', marker='x', s=200, linewidths=3)
plt.title(f'Кластеризація методом зсуву середнього (кількість кластерів: {n_clusters})')
plt.xlabel('Ознака 1')
plt.ylabel('Ознака 2')
plt.show()

print(f"Кількість кластерів: {n_clusters}")
print("Центри кластерів:")
print(cluster_centers)