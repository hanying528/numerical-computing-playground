from PIL import Image

import numpy as np
from matplotlib import pyplot as plt


def load_image(fp, gray=False):
    """Load image as np array"""
    if not gray:
        img = Image.open(fp)
    else:
        img = Image.open(fp).convert('L')
    img_arr = np.asarray(img, dtype="int32")
    # crop the image to square
    img_arr = img_arr[50:250, 50:250]
    return img_arr


def save_image(arr, out):
    """Save np arrays as image"""
    img = Image.fromarray(np.asarray(np.clip(arr, 0, 255),
                                     dtype="uint8"), "L")
    img.save(out)
    print(f"Saved image to {out}")


def display_img(arr, title=None, save_to=None):
    """Display the given numpy arrays as image"""
    if title:
        plt.title(title)
    plt.imshow(arr)
    plt.show()
    if save_to:
        save_image(arr, save_to)


def compress(U, S, Vt, r, save_to=None):
    """Recreate an image using the given rank
    Reference: https://datahacker.rs/009-the-singular-value-decompositionsvd-illustrated-in-python/
    """
    print(f"Rank {r} approximation")
    assert r <= S.shape[0], f"Rank cannot exceed {S.shape[0]}"
    img_r = U[:, :r].dot(S[:r, :r]).dot(Vt[:r, :])
    display_img(img_r, f"A rank-{r} approximation")
    if save_to:
        save_image(img_r, save_to)
