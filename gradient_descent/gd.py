import numpy as np

from utils import loss_func, gradient


def gradient_descent(x, y, w_init=None, eta=0.01, max_iter=500, tolerance=1e-03,
                     seed=None, debug=False):
    """Iteratively update b based on the learning rate and the gradient,
    return the last position found either when it reaches the max iteration number or
    the improvement is smaller than specified tolerance level, whichever comes first.

    Parameters
    ----------
    x : numpy.ndarray
    y : numpy.ndarray
    w_init : numpy.ndarray, optional
        the initial guess of w, if None, randomly initialize it
    eta : float, optional
        the constant learning rate, default is 0.1
    max_iter : int, optional
        the max number of iteration, default is 500
    tolerance : float, optional
        stop iteration if improvement is smaller than this threshold, default is 1e-03
    seed : int, optional
        the random seed used to take random numbers
    debug : bool, optional
        print debugging message if set to True, default is False

    Returns
    -------
    b : numpy array
    losses : numpy.ndarray
    """
    i = 0
    k = x.shape[1]  # number of features

    # randomly initialize weights if not already
    np.random.seed(seed=seed)
    w = np.random.randn(k, 1) if not w_init else w_init

    losses = np.zeros(max_iter)
    ea = 1  # assume ea starts with 100%

    while i < max_iter and ea > tolerance:
        y_hat = x.dot(w)

        diff = -eta * gradient(x, y, y_hat)
        w += diff  # update w

        losses[i] = loss_func(y, y_hat)
        ea = np.abs(1 - losses[i-1] / losses[i])

        if debug:
            print(f"Loss at iteration {i}: {losses[i]}; approximate relative error: {ea}")

        i += 1

    print(f"[GD method] Final loss: {losses[i-1]}\nEnd at {i}th iteration")
    return w, losses[:i]
