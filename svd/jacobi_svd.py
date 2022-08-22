# Implementation of Jacobi SVD based on http://www.math.pitt.edu/~sussmanm/2071Spring08/lab09/index.html#Exercise4
# One-sided Jacobi algorithm for SVD
# lawn15: Demmel, Veselic, 1989,
# Algorithm 4.1, p. 32

import numpy as np


def jacobi_svd(A, max_iter=500):
    """ One-sided Jacobi method for solving svd.

    Iterative directly on input matrix without reducing to bidiagonal form.

    Parameters
    ----------
    A : np.ndarray

    Returns
    -------
    U, S, V: np.ndarray where A = U * S * Vt
    """
    tolerance = 1e-6
    _, n = A.shape
    U = np.copy(A)
    V = np.identity(n)
    converge = tolerance + 1
    iter = 0
    while converge > tolerance and iter < max_iter:
        converge = 0
        for j in range(1, n):
            for i in range(j):
                # compute alpha, beta and gamma
                alpha = np.sum(U[:, i] ** 2)
                beta = np.sum(U[:, j] ** 2)
                gamma = np.dot(U[:, i], U[:, j])
                converge = max(converge, abs(gamma)/np.sqrt(alpha * beta))

                # for efficiency
                if np.isclose(beta, alpha, atol=tolerance)\
                        or np.isclose(gamma, 0):
                    continue

                # compute Jacobi rotation
                zeta = (beta - alpha) / (2 * gamma)
                tangent = np.sign(zeta) / (abs(zeta) + np.sqrt(1 + zeta ** 2))
                cosine = 1 / np.sqrt(1 + tangent ** 2)
                sine = cosine * tangent
                # print("cos:", cosine, "sine:", sine)
                # update columns i and j of U
                U_i = U[:, i].copy()
                U[:, i] = cosine * U_i - sine * U[:, j]
                U[:, j] = sine * U_i + cosine * U[:, j]

                # update matrix Vt of right singular value
                V_i = V[:, i].copy()
                V[:, i] = cosine * V_i - sine * V[:, j]
                V[:, j] = sine * V_i + cosine * V[:, j]
            # end for
        # end for
        iter += 1

        if np.isclose(beta, alpha, atol=tolerance):
            # avoid infinite loop
            break
    # end while

    # Calculate singular values - norms of the columns of U
    sing_vals = np.empty(shape=n)
    for k in range(n):
        sing_vals[k] = np.linalg.norm(U[:, k])
        U[:, k] /= sing_vals[k]

    # sort singular values in descending order
    idx = np.argsort(-sing_vals)
    S = np.diag(sing_vals[idx])

    # permute U and V based on indices
    V_new = V.copy()
    for i in range(n):
        V_new[:, i] = V[:, idx[i]]
    return U[:, idx], S, V_new.T


def test_jacobi(debug=False):
    """Only for testing jacobi svd implementation"""
    A = np.arange(1, 13.0).reshape(4, 3)
    A_test = np.copy(A)
    if debug:
        print(A)

    U, S, Vt = jacobi_svd(A)

    if debug:
        print("U:\n", U)
        print("S:\n", S)
        print("Vt:\n", Vt)
        print("U * S * Vt:\n", U.dot(S).dot(Vt))

    assert np.allclose(U.dot(S).dot(Vt), A_test)
