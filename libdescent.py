import numpy as np


def sse(a, x, y):
    return np.sum((np.dot(x, a) - y)**2)


def gradient(coefs, x, y):
    g = 0
    for i in range(x.shape[0]):
        g += 2 * (np.dot(x[i], coefs) - y[i]) * x[i]
    return g

def descent(x, y, eps = 1e-3, steps = 1000):
    n, m = x.shape[0], x.shape[1]
    a = np.zeros(m)
    alpha = 1.0 

    for i in range(1, steps+1):
        g = gradient(a, x, y)
        dist = np.linalg.norm(g)

        if dist < (n**(.5))*eps:
            break

        else:
            if sse(a - alpha * g, x ,y) < sse(a, x, y):
                a = a - alpha * g
                alpha *= 2
            else:
                alpha /= 2

    return (a, i)

