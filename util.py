import numpy as np

TEST_SIZE = 1000 * 1

SUPPORTED_DNNS = ['ReLU_Softmax_AdamOptimizer',
                  'Linear_Softmax_GradientDescentOptimizer',
                  'ReLU_Softmax_AdTraining',
                  'Inception']


def flip_black_white(x):
    # flip white and black color
    flip_matrix = np.ones(x.shape)
    x_flipped = flip_matrix - x
    return x_flipped