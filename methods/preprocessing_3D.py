
from scipy import ndimage

LABEL2ONEHOT = {"young": [1.0, 0.0], "aged": [0.0, 1.0]}


def resize_volume(img, new_size):
    """Resize a 3D volume using ndimage.zoom"""

    # Compute the resizing factors
    factor_vec = [(new / old) for new, old in zip(new_size, img.shape)]

    # Return the actual resize
    return ndimage.zoom(img, factor_vec, order=1)
