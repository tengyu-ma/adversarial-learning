import cifar10
import cifar10_train
import cifar10_eval
import cifar10_adversarial
from copy import copy
from settings import *


def train_with_original_data():
    global FLAGS
    FLAGS.image_size = 32
    cifar10_train.main()


def generate_images_size24():
    FLAGS.image_size = 32
    cifar10_adversarial.generate_images_size24()


def eval():
    global FLAGS
    FLAGS.image_size = 24
    FLAGS.eval_data_set = "test_batch_size24.bin"
    cifar10_eval.evaluate()


if __name__ == '__main__':
    # train_with_original_data()
    generate_images_size24()
    # eval()
