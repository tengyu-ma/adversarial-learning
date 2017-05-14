import cifar10
import cifar10_train
from copy import copy
from settings import *


def train_with_orginal_data():
    global FLAGS
    org_image_size = FLAGS.image_size
    FLAGS.image_size = 32
    cifar10_train.main()
    FLAGS.image_size = org_image_size
    print(FLAGS.image_size)

def eval():
    pass


if __name__ == '__main__':
    # train_with_orginal_data()
    eval()
