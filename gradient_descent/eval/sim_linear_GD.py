import time
import numpy as np
from gd import gradient_descent
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
def run_GD():
    # Gradient descent
    # start = time.time()
    # w_gd, losses_gd = gradient_descent(X, Y, seed=seed, eta=0.03, tolerance=1e-06, max_iter=1000)
    # print(w_gd)
    # print(f"Total time used: {time.time() - start} seconds")

    start = time.time()
    w_gd, losses_gd = gradient_descent(X, Y, seed=seed, eta=0.03, tolerance=1e-06, max_iter=1000)
    print(f"Total time used: {time.time() - start} seconds")
    Y_hat_gd = X.dot(w_gd)
    print(w_gd)

if __name__ == '__main__':
    run_GD()
