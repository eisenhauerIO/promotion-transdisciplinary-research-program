import matplotlib.pyplot as plt
import numpy as np


def increasing(x, a):
    return a * x ** 2


def decreasing(x, a, n):
    a = 1
    n = 0.4
    return 1 - (x / a) ** n


x = np.linspace(0, 1, 1000)

y_incr = np.apply_along_axis(increasing, 0, x, *[0.5])
y_decr = np.apply_along_axis(decreasing, 0, x, *[1, 0.4])


fig, ax = plt.subplots()
ax.plot(x, y_incr, label="Uncertainty propagation")
ax.plot(x, y_decr, label="Model limitations")
ax.plot(x, y_decr + y_incr, label="Model Error")


ax.set_ylim(-0.1, 1)
ax.legend()
ax.set_xlabel("Model complexity")
ax.set_ylabel("Model error")

ax.set_yticklabels([])
ax.set_xticklabels([])


fig.savefig("fig-illustration-complexity")
