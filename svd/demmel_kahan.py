# Implementation of Demmel Kahan SVD based on http://www.math.pitt.edu/~sussmanm/2071Spring08/lab09/index.html#Exercise4
import sys
import numpy as np

from householder import householder_bidiag_reduction


def rot(f, g):
    """Computes the cosine c and sine s of a rotation angle that satisfies the following condition.
    [[c, s],  [[f],   [[r],
     [-s, c]]  [g]] =  [0]]
    """
    if f == 0:
        c = 0
        s = 1
        r = g
    elif abs(f) > abs(g):
        t = g / f
        t1 = np.sqrt(1 + t ** 2)
        c = 1 / t1
        s = t * c
        r = f * t1
    else:
        t = f / g
        t1 = np.sqrt(1 + t ** 2)
        s = 1 / t1
        c = t * s
        r = g * t1
    return c, s, r


def test_rot():
    """For testing the rot() implementation above"""
    f = 1
    g = 0
    c, s, r = rot(f, g)
    assert c == 1 and s == 0 and r == 1


def msweep(B):
    """Sweep through the bidiagonal matrix B repeatedly.
    Demmel & Kahan zero-shift QR downward sweep
    """
    n = B.shape[1]

    for i in range(n-1):
        c, s, r = rot(B[i, i], B[i, i+1])

        # construct the rotation matrix Q
        Q = np.identity(n)
        Q[i:i+2, i:i+2] = np.array([[c, s], [-s, c]])
        B = B.dot(Q.T)
        c, s, r = rot(B[i, i], B[i+1, i])

        # construct the rotation matrix Q again
        Q = np.identity(n)
        Q[i:i+2, i:i+2] = np.array([[c, s], [-s, c]])
        B = Q.dot(B)
    return B


def vsweep(d, e):
    """Sweep through the diagonal and superdiagonal d, e repeatedly.
    Demmel & Kahan zero-shift QR downward sweep
    """
    n = len(d)
    c, c_old = 1, 1
    for i in range(n-1):
        c, s, r = rot(c * d[i], e[i])

        if i > 0:
            e[i-1] = r * s_old
        c_old, s_old, d[i] = rot(c_old * r, d[i + 1] * s)
    h = c * d[n-1]
    e[n-2] = h * s_old
    d[n-1] = h * c_old
    return d, e


def test_sweep_methods():
    A = np.arange(1, 10.0).reshape(3, 3)
    _, B, _ = householder_bidiag_reduction(A)
    B_m = B.copy()
    # Sweep on Matrix
    B_msweep = msweep(B_m)

    # Sweep on d and e
    B_v = B.copy()
    n = B_v.shape[1]
    d = B_v.diagonal()
    e = B_v[:n - 1, 1:].diagonal()  # superdiagonal
    d.setflags(write=1)
    e.setflags(write=1)
    d, e = vsweep(d, e)

    # Update B based on new d, e
    B_vsweep = B.copy()
    np.fill_diagonal(B_vsweep, d)
    np.fill_diagonal(B_vsweep[:n - 1, 1:], e)

    assert np.allclose(B_msweep, B_vsweep)


def demmel_kahan_svd(B):
    """Apply Demmel Kahan svd on a n * n bidiagonal matrix B"""
    tolerance = 1e-6
    n = B.shape[1]
    max_iter = 100 * n ** 2
    d = B.diagonal()
    e = B[:n - 1, 1:].diagonal()  # superdiagonal
    d.setflags(write=1)
    e.setflags(write=1)

    # determine convergence criterion
    l = np.abs(d)
    ls = np.empty(n-1)
    for j in range(n-2, -1, -1):
        ls[j] = np.abs(d[j]) * l[j+1] / (l[j+1] + np.abs(e[j]))
    mu = np.empty(n-1)
    mu[0] = np.abs(d[0])
    for j in range(n-2):
        mu[j+1] = np.abs(d[j + 1]) * (mu[j] / (mu[j] + np.abs(e[j])))
    ls_min = np.min(ls)
    mu_min = np.min(mu)

    sigma_lower = min(ls_min, mu_min)
    threshold = max(tolerance * sigma_lower, max_iter * sys.float_info.min)
    print("threshold:", threshold)

    i_upper = n - 2
    i_lower = 0
    for iter in range(max_iter):
        for i in range(i_upper, -1, -1):
            i_upper = i
            if np.abs(e[i]) > threshold:
                break
        # end for
        # how many zeros are near the top
        j = i_upper
        for i in range(i_lower, i_upper):
            if np.abs(e[i]) > threshold:
                j = i
                break
        i_lower = j
        if i_lower == i_upper or i_upper < i_lower:
            # when condition met, return the updated d and e
            d = np.sort(np.abs(d))[::-1]
            return d
        # otherwise, sweep
        d[i_lower: i_upper + 1], e[i_lower: i_upper] = vsweep(d[i_lower: i_upper + 1], e[i_lower: i_upper])


def test_demmel_kahan_svd():
    A = np.arange(1, 17.0).reshape(4, 4)
    U, B, Vt = householder_bidiag_reduction(A)
    n = B.shape[1]
    d_dk = demmel_kahan_svd(B)
    S_dk = np.identity(n)
    np.fill_diagonal(S_dk, d_dk)

    # singular values from numpy
    u_np, s_np, vt_np = np.linalg.svd(A)
    assert np.allclose(np.diag(s_np), S_dk)
