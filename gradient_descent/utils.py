import numpy as np


def loss_func(y, y_hat):
    """Return the mean square error given y and predicted y"""
    return (1/(2*len(y))) * np.sum(np.square(y_hat - y))


def gradient(x, y, y_hat):
    """Return the gradient of the cost function given x, y and predicted y"""
    return (1/len(y)) * x.T.dot(y_hat - y)
