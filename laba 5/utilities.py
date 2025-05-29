import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y, title=''):
    # Визначаємо межі для сітки
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    
    # Створюємо сітку з кроком 0.01
    mesh_step_size = 0.01
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                               np.arange(min_y, max_y, mesh_step_size))
    
    # Передбачаємо вихідні значення для всіх точок сітки
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)
    
    # Створюємо графік і візуалізуємо результати
    plt.figure()
    plt.title(title)
    
    # Відображаємо області класифікації за допомогою colormesh
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
    
    # Відображаємо точки даних
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black',
                linewidth=1, cmap=plt.cm.Paired)
    
    # Налаштовуємо межі графіка
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    
    # Налаштовуємо мітки на осях
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))
    
    plt.show()