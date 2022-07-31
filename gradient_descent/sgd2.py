import numpy as np

from utils import loss_func, gradient


def stochastic_gradient_descent(x, y, w_init=None, eta=0.01, max_iter=500, tolerance=1e-03,
                                seed=None, debug=False):
    """Similar to gradient descent, except that it calculates gradient on the random observations
    instead of all the observations one by one.

    Parameters
    ----------
    x : numpy.ndarray
    y : numpy.ndarray
    w_init : numpy.ndarray, optional
        initial guess of w, if None, randomly initialize it
    eta : float, optional
        learning rate, default is 0.01
    max_iter : int, optional
        max number of iteration, default is 500
    tolerance : float, optional
        stop iteration if moving step is smaller than this threshold, default is 1e-03
    seed : int, optional
        random seed used to take random numbers
    debug : bool, optional
        print debugging message if set to True, default is False

    Returns
    -------
    b : numpy.ndarray
    losses : numpy.ndarray
    """
    i = 0
    k = x.shape[1]  # number of features
    n = len(y)

    # randomly initialize weights if not already
    np.random.seed(seed=seed)
    w = np.random.randn(k, 1) if not w_init else w_init

    losses = np.zeros(max_iter)
    ea = 1  # assume ea starts with 100%

    count = 0
    while i < max_iter and ea > tolerance:
        loss_i = 0.
        shuff_indx = np.random.permutation(n)
        for _ in range(n):
            x_ = x[shuff_indx][_, ].reshape(1, k)
            y_ = y[shuff_indx][_].reshape(1, 1)
            y_hat = np.dot(x_, w)

            diff_ = -eta * gradient(x_, y_, y_hat)
            w += diff_

            # accumulate loss
            loss_i += loss_func(y_, y_hat)
            count += 1

        losses[i] = loss_i / n
        ea = np.abs(1 - losses[i-1] / losses[i])

        if debug:
            print(f"Loss at iteration {i}: {losses[i]}; approximate relative error: {ea}; count: {count}")

        i += 1

    print(f"[SGD method] Final loss: {losses[i-1]}\nEnd at {i}th iteration")
    return w, losses[:i]
