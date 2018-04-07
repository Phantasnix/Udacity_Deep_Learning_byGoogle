"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np
from math import exp

def cal_1D_softmax(x):
    exped = [exp(v) for v in x]
    sum_ = sum(exped)
    return np.array([v / sum_ for v in exped])

def cal_2D_softmax(x):
    x_t = x.transpose()
    transposed = [cal_1D_softmax(col_t) for col_t in x_t]
    return np.array(transposed).transpose()

def softmax_(x):
    """Compute softmax values for each sets of scores in x."""
    dim = 1
    try:
        dim = x.ndim
    except AttributeError:
        return cal_1D_softmax(x)
    if dim == 1:
        return cal_1D_softmax(x)
    elif dim == 2:
        return cal_2D_softmax(x) 
       
def softmax(x):
    '''One line work if I learn how to play NumPy well......'''
    return np.exp(x) / np.sum(np.exp(x), axis = 0)

test = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])
print np.sum(test, axis = 1)

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
