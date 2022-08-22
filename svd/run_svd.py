from pathlib import Path
from time import time

import numpy as np

from demmel_kahan import demmel_kahan_svd
from householder import householder_bidiag_reduction
from jacobi_svd import jacobi_svd
from utils import compress, display_img, load_image


def main():
    # load a sample image
    fp = Path(__file__).parent / 'img_sample/Ragdoll_20.jpg'
    bs = load_image(fp)
    display_img(bs)

    # convert to gray scale for simplicity
    bs_gray = load_image(fp, gray=True).astype(np.float32)
    display_img(bs_gray, save_to="gray_scale_bs.jpg")

    rank = 10

    # numpy.linalg.svd
    s = time()
    print("numpy.linalg.svd:")
    u_svd, s_svd, vt_svd = np.linalg.svd(bs_gray)
    print("Total time took (seconds):", time() - s)
    compress(u_svd, np.diag(s_svd), vt_svd, rank,
             save_to=f'compressed_img_np_svd_rank{rank}.jpg')

    # Demmel Kahan method
    s = time()
    print("Demmel Kahan method:")
    _, B, _ = householder_bidiag_reduction(bs_gray)
    d_dk = demmel_kahan_svd(B)
    n = len(d_dk)
    s_dk = np.identity(n)
    np.fill_diagonal(s_dk, d_dk)
    print("Total time took (seconds):", time() - s)

    # Jacobi svd method
    s = time()
    print("Jacobi method:")
    u_jacobi, s_jacobi, vt_jacobi = jacobi_svd(bs_gray)
    print("Total time took (seconds):", time() - s)
    compress(u_jacobi, s_jacobi, vt_jacobi, rank,
             save_to=f'compressed_img_jacobi_svd_rank{rank}.jpg')


if __name__ == '__main__':
    main()
