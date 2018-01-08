import libdescent as ld, numpy as np

x = np.array([[1, 0, 1], [1, 1, 2], [1, 0.5, 2]])
coefs = np.array([1, 0.3, 0.5])
y = np.inner(x, coefs)

ld.descent(x, y)

ld.descent(x, y, eps = 1e-2)

ld.descent(x, y, steps = 50)