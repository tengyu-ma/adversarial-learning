import cifar10
import cifar10_train
import cifar10_eval
import cifar10_adversarial
import autoencoder_run
import generate_bin_from_npy
import io_binary
import sys
from settings import *


def train_with_original_data():
    global FLAGS
    FLAGS.use_processed_data = False
    cifar10_train.main()


def train_with_processed_data():
    global FLAGS
    # io_binary.processing_images_without_noise()
    FLAGS.use_processed_data = True
    cifar10_train.main()


def generate_images_size24(i=0):
    global FLAGS
    FLAGS.get_single_label = True
    FLAGS.use_processed_data = False  # to set
    if i == 0:
        FLAGS.eval_data_set = "test_batch.bin"
    else:
        FLAGS.eval_data_set = "data_batch_%d.bin" % i
    filename = FLAGS.eval_data_set.split('.')[0]
    if FLAGS.use_processed_data:
        filename += '_processed'
        FLAGS.eval_data_set = filename + '.bin'
    filename += '_size24.bin'
    FLAGS.generate_data_set = filename
    cifar10_adversarial.generate_images_size24()


def generate_images_with_noise(i=0):
    global FLAGS
    FLAGS.image_size = 24
    FLAGS.get_single_label = True
    FLAGS.use_processed_data = False
    if i == 0:
        FLAGS.eval_data_set = "test_batch.bin"
    else:
        FLAGS.eval_data_set = "data_batch_%d.bin" % i
    filename = FLAGS.eval_data_set.split('.')[0]
    if FLAGS.use_processed_data:
        filename += '_processed'
    filename += '_size24'
    FLAGS.eval_data_set = filename + '.bin'
    filename = filename + '_eps%d' % EPS
    FLAGS.generate_data_set = filename
    cifar10_adversarial.generate_images_with_noise()
    FLAGS.get_single_label = False


def evaluate():
    global FLAGS
    FLAGS.image_size = 24
    # FLAGS.eval_data_set = "test_batch_size24.bin"
    # FLAGS.eval_data_set = 'test_batch_size24_eps%d_noise.bin' % EPS
    FLAGS.eval_data_set  = 'test_batch_size24_eps%d_noise_after_cae.bin' % EPS
    cifar10_eval.evaluate()


def show_images_with_noise():
    global FLAGS
    FLAGS.image_size = 24
    FLAGS.eval_data_set = "test_batch_size24.bin"
    cifar10_adversarial.show_images_with_noise()


def process_image_with_autoencoder():
    global FLAGS
    FLAGS.autoencoder_test_set = 'test_batch_size24_eps%d_noise.bin' % EPS
    autoencoder_run.process_image_with_autoenccoder()
    generate_bin_from_npy.main()


if __name__ == '__main__':
    # train_with_original_data()
    # generate_images_size24(5)
    # generate_images_with_noise()
    # process_image_with_autoencoder()
    # show_images_with_noise()
    evaluate()
