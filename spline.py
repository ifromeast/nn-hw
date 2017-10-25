from scipy import interpolate
import numpy as np
from math import sin, exp, pi, cos
import matplotlib.pyplot as plt


length = 500
y = np.zeros([length])
y[0] = 1
y[1] = 1
for i in range(length):
    y[i] = 0.3 * y[i - 1] + 0.6 * y[i - 2] + 0.6 * sin(
        pi * cos(2 * pi * i / 250)) + 0.9 * exp(-y[i - 1] ** 2)

x = np.linspace(1, length, length)

# x = np.transpose([x])
# y = np.transpose([y])

# nearest, zero, quadratic, cubic
f = interpolate.interp1d(x, y, kind='quadratic')
y_predict = f(x)

plt.figure(1)
plt.plot(x, y)
plt.figure(2)
plt.plot(x, y_predict)
plt.show()
