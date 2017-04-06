import numpy as np

TEST_SIZE = 1000 * 1

def flip_black_white(x):
    # flip white and black color
    flip_matrix = np.ones(x.shape)
    x_flipped = flip_matrix - x
    return x_flipped