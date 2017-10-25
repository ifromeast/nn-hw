import numpy as np
import matplotlib.pyplot as plt
from math import sin, exp, pi, cos
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

length = 500
y = np.zeros([length])
y[0] = 1
y[1] = 1
for i in range(length):
    y[i] = 0.3 * y[i - 1] + 0.6 * y[i - 2] + 0.6 * sin(
        pi * cos(2 * pi * i / 250)) + 0.9 * exp(-y[i - 1] ** 2)

x = np.linspace(1, length, length)
x = np.transpose([x])
y = np.transpose([y])


qf = PolynomialFeatures(degree=8)
qModel = LinearRegression()
qModel.fit(qf.fit_transform(x), y)
y_predict = qModel.predict(qf.transform(x))

plt.plot(x, y)
plt.plot(x, y_predict)
plt.show()
