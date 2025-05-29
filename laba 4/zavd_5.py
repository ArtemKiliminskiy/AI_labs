import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Генерація даних
m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Візуалізація даних
plt.scatter(X, y, color='blue', label='Дані')
plt.title('Генерація даних')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

# Поліноміальна регресія
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)
y_poly_pred = lin_reg_poly.predict(X_poly)

# Виведення параметрів моделей
print(f"Коефіцієнти лінійної регресії: {lin_reg.coef_}")
print(f"Перехоплення лінійної регресії: {lin_reg.intercept_}")
print(f"Коефіцієнти поліноміальної регресії: {lin_reg_poly.coef_}")
print(f"Перехоплення поліноміальної регресії: {lin_reg_poly.intercept_}")

# Візуалізація результатів
plt.scatter(X, y, color='blue', label='Дані')
plt.plot(X, y_lin_pred, color='red', label='Лінійна регресія')

X_sorted = np.sort(X, axis=0)
X_poly_sorted = poly_features.transform(X_sorted)
y_poly_pred_sorted = lin_reg_poly.predict(X_poly_sorted)

plt.plot(X_sorted, y_poly_pred_sorted, color='green', label='Поліноміальна регресія')
plt.title('Лінійна і поліноміальна регресії')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Виведення рівнянь
print(f"Лінійне рівняння: y = {lin_reg.intercept_[0]} + ({lin_reg.coef_[0][0]}) * X")
print(f"Поліноміальне рівняння: y = {lin_reg_poly.intercept_[0]} + ({lin_reg_poly.coef_[0][0]}) * X + ({lin_reg_poly.coef_[0][1]}) * X^2")