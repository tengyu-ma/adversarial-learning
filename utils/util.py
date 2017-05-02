import os

TEST_SIZE = 1000 * 1

SUPPORTED_DNNS = ['ReLU_Softmax_AdamOptimizer',
                  'Linear_Softmax_GradientDescentOptimizer',
                  'ReLU_Softmax_AdTraining',
                  'Inception',
                  'CIFAR-10']

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))[:-len("\\utils")]
ImageNet_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))[:-len("\\utils")], "data", "imagenet")