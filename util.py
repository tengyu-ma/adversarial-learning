import numpy as np


def flip_black_white(x):
    # flip white and black color
    flip_matrix = np.ones(x.shape)
    x_flipped = flip_matrix - x
    return x_flipped