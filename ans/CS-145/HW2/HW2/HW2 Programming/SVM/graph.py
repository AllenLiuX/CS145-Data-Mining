import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return -3.94951284322 * t + 6.91260702687

x1 = [0.52, 0.91, -1.48, 0.01, -0.46, 0.41, 0.53, -1.21, -0.39, -0.96, 2.46, 3.05, 2.2, 1.89, 4.51, 3.06, 3.16, 2.05, 2.34, 2.94]
x2 = [-1, 0.32, 1.23, 1.44, -0.37, 2.04, 0.77, -1.1, 0.96, 0.08, 2.59, 2.87, 3.04, 2.64, -0.52, 1.3, -0.56, 1.54, 0.72, 0.13]

print(len(x1))
print(len(x2))

colors = ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']

line_x = [-2, -1, 0, 1, 2, 3, 4, 5]
line_y = []

for i in line_x:
    line_y.append(f(i))

plt.xlabel("x1")
plt.ylabel("x2")
plt.scatter(x1, x2, c=colors)
plt.plot(line_x, line_y, 'r--')
plt.show()
