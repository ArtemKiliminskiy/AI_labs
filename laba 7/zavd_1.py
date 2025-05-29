import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = np.loadtxt('data_clustering.txt', delimiter=',')

plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], color='blue')
plt.title('Входные данные')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
kmeans.fit(data)

x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(12, 8))
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap='viridis', aspect='auto', origin='lower')

plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='x', s=200, linewidths=3)
plt.title('Кластеризация методом k-средних')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.show()

print("Центри кластерів:")
print(kmeans.cluster_centers_)