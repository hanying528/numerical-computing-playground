import time
import numpy as np


seed = 0
np.random.seed(seed=seed)
n = 1_000_000


@profile
def setup():
    data = np.load(f'sim_multi_linear_data_{n}.npy')
    X = data[:, :-1]
    Y = data[:, -1].reshape(n, 1)
    return X, Y


X, Y = setup()

@profile
def run_OLS():
    # Solve normal equation
    start = time.time()
    w_normal = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)  # normal equation
    print(w_normal)
    print(f"Total time used: {time.time() - start} seconds")


if __name__ == '__main__':
    run_OLS()
