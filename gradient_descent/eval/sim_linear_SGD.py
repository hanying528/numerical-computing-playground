import time

import numpy as np

from sgd import stochastic_gradient_descent
from utils import loss_func

seed = 0
np.random.seed(seed=seed)
n = 100_000


@profile
def setup():
    data = np.load(f'sim_linear_data_{n}.npy')
    X = data[:, :-1]
    Y = data[:, -1].reshape(n, 1)
    return X, Y


X, Y = setup()


@profile
def run_SGD():
    # Stochastic gradient descent
    # start = time.time()
    # w_sgd, losses_sgd = stochastic_gradient_descent(X, Y, seed=seed, eta=0.01, tolerance=1e-06)
    # print(f"Total time used: {time.time() - start} seconds")
    # print(w_sgd)

    start = time.time()
    w_sgd, losses_sgd = stochastic_gradient_descent(X, Y, seed=seed, eta=0.001, tolerance=1e-03)
    print(w_sgd)
    print(f"Total time used: {time.time() - start} seconds")


if __name__ == '__main__':
    run_SGD()
