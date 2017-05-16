import cifar10
import cifar10_train
import cifar10_eval
import cifar10_adversarial
import autoencoder_run
from copy import copy
from settings import *


def train_with_original_data():
    cifar10_train.main()


def generate_images_size24():
    global FLAGS
    FLAGS.get_single_label = True
    FLAGS.eval_data_set = "test_batch.bin"
    cifar10_adversarial.generate_images_size24()
    FLAGS.get_single_label = False


def generate_images_with_noise():
    global FLAGS
    FLAGS.image_size = 24
    FLAGS.get_single_label = True
    FLAGS.eval_data_set = "test_batch_size24.bin"
    cifar10_adversarial.generate_images_with_noise()
    FLAGS.get_single_label = False


def evaluate():
    global FLAGS
    FLAGS.image_size = 24
    FLAGS.eval_data_set = "test_batch_size24.bin"
    cifar10_eval.evaluate()


def show_images_with_noise():
    global FLAGS
    FLAGS.image_size = 24
    FLAGS.eval_data_set = "test_batch_size24.bin"
    cifar10_adversarial.show_images_with_noise()


def process_image_with_autoencoder():
    autoencoder_run.process_image_with_autoenccoder()


if __name__ == '__main__':
    # train_with_original_data()
    generate_images_size24()
    # generate_images_with_noise()
    # process_image_with_autoencoder()
    # show_images_with_noise()
    # evaluate()