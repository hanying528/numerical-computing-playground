import numpy as np


def householder_vector(a):
    """Compute Householder vector for the given column

    Parameters
    ----------
    a : np.ndarray

    Returns
    -------
    v : np.ndarray
    tau : float
    """
    v = np.copy(a)
    a0 = a[0]
    s = np.power(np.linalg.norm(a[1:]), 2)

    if s == 0:
        tau = 0
    else:
        norm = np.sqrt(a0 ** 2 + s)
        if a0 <= 0:
            v[0] = a0 - norm
        else:
            v[0] = -s / (a0 + norm)
        tau = 2 * v[0] ** 2 / (s + v[0] ** 2)
        v /= v[0]
    return v, tau


def householder_bidiag_reduction(A):
    """Reduce the given matrix A to bidiagonal form

    Parameters
    ----------
    A : np.ndarray

    Returns
    -------
    U, B, Vt : np.ndarray where U * B * Vt = A
    """
    m, n = A.shape
    if m < n:
        raise ValueError("Number of rows must be greater than or equal to number of columns")
    U, Vt = np.identity(m), np.identity(n)

    for i in range(n):
        # compute householder vector
        v, tau = householder_vector(A[i:, i])

        # update A inplace
        A[i:, i:] = (np.identity(m - i) - tau * np.outer(v, v)) @ A[i:, i:]

        # update U
        Q = np.identity(m)
        Q[i:, i:] -= tau * np.outer(v, v)
        U = U @ Q

        if i <= n - 2:
            # take the transpose and apply householder again
            v, tau = householder_vector(A[i, i + 1:].T)
            A[i:, i+1:] = A[i:, i+1:] @ (np.identity(n - (i + 1)) - tau * np.outer(v, v))
            # update Vt
            P = np.identity(n)
            P[i+1:, i+1:] -= tau * np.outer(v, v)
            Vt = P @ Vt

    return U, A, Vt


def test_householder(debug=False):
    """Only for testing householder algorithm implementation"""
    A = np.arange(1, 13.0).reshape(4, 3)
    A_test = np.copy(A)
    if debug:
        print(A)

    U, B, Vt = householder_bidiag_reduction(A)

    if debug:
        print("U:\n", U)
        print("B:\n", B)
        print("Vt:\n", Vt)

    assert np.allclose(U.dot(B).dot(Vt), A_test)
