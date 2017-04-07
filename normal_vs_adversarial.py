from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import toimage
from pylab import *

import tensorflow as tf
import adversarial_generator as ag
import learning_strategy
import denoising_strategy as ds
import util

import numpy as np
import logging
import random
import os
import cv2


class NormalVsAdversarial:
    def __init__(self, data_set_name='MNIST_data', iter=1000):
        self.data = input_data.read_data_sets(data_set_name, one_hot=True)  # dataset
        self.sess = tf.Session()  # tensorflow session
        self.x = tf.placeholder(tf.float32, shape=[None, 784])  # true training data
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])  # true value
        self.y = None  # predicted value by learning
        self.cost_function = None  # cost function for a specific learning model
        self.iter = iter  # iteration number for training
        self.dropout = False  # dropout in order to avoid overfitting
        self.keep_prob = tf.placeholder(tf.float32)  # drop out probability of the network
        self.adv_method = 'fast_gradient_sign_method'
        self.denoised_method = 'threshold_method'

    def run_training(self, learning_strategy_name):
        # run training based on the learning_strategy_name
        print('===== %s %d iteration =====' % (learning_strategy_name, self.iter))
        logging.info('===== %s %d iteration =====' % (learning_strategy_name, self.iter))
        learning_model = getattr(learning_strategy, learning_strategy_name)
        # initialize local variables
        data, sess, x, y_, keep_prob, iter = self.data, self.sess, self.x, self.y_, self.keep_prob, self.iter
        self.y, self.cost_function = learning_model(data, self.sess, x, y_, keep_prob, iter, 0)
        # save the model
        save_path = ".%scheckpoint%s%s.ckpt" % (os.sep, os.sep, learning_strategy_name)
        saver = tf.train.Saver()
        save_path = saver.save(NvA.sess, save_path)
        print("[+] Model saved in file: %s" % save_path)

    def restore_network(self, learning_strategy_name):
        logging.info('===== %s %d iteration =====' % (learning_strategy_name, self.iter))
        learning_model = getattr(learning_strategy, learning_strategy_name)
        data, sess, x, y_, keep_prob, iter = self.data, self.sess, self.x, self.y_, self.keep_prob, self.iter
        self.y, self.cost_function = learning_model(data, self.sess, x, y_, keep_prob, iter, 1)

    def normal_test(self):
        # normal case
        data, sess, x, y_, keep_prob = self.data, self.sess, self.x, self.y_, self.keep_prob
        x_test_normal = data.test.images[0:util.TEST_SIZE]
        y_test_normal = data.test.labels[0:util.TEST_SIZE]
        accuracy, avg_confidence = self.evaluate(x_test_normal, y_test_normal)
        print('* Normal Test\nAccuracy\tConfidence')
        print('%s\t\t%s' % (accuracy, avg_confidence))
        logging.info('%s\t\t%s\t\t%s' % ('normal', accuracy, avg_confidence))

    def adversarial_test(self, epsilon=0.1):
        # adversarial case
        data, sess, x, y_, keep_prob = self.data, self.sess, self.x, self.y_, self.keep_prob
        x_test_normal = data.test.images[0:util.TEST_SIZE]
        y_test_normal = data.test.labels[0:util.TEST_SIZE]
        x_test_adversarial, y_test_adversarial, noise = self.adversarialize(x_test_normal, y_test_normal, epsilon)
        accuracy, avg_confidence = self.evaluate(x_test_adversarial, y_test_normal)
        print('* Adversarial Test\nAccuracy\tConfidence')
        print('%s\t\t%s' % (accuracy, avg_confidence))
        logging.info('%s\t%s\t\t%s\t%s' % ('adversarial', accuracy, avg_confidence, epsilon))

    def adversarial_test_denoised(self, epsilon=0.1, thres=0.5):
        data, sess, x, y_, keep_prob = self.data, self.sess, self.x, self.y_, self.keep_prob
        x_test_normal = data.test.images[0:util.TEST_SIZE]
        y_test_normal = data.test.labels[0:util.TEST_SIZE]
        x_test_adversarial, y_test_adversarial, noise = self.adversarialize(x_test_normal, y_test_normal, epsilon)
        accuracy, avg_confidence = self.evaluate(x_test_adversarial, y_test_normal)
        print('* Adversarial Test\nAccuracy\tConfidence')
        print('%s\t\t%s' % (accuracy, avg_confidence))
        logging.info('%s\t%s\t\t%s\t%s' % ('adversarial', accuracy, avg_confidence, epsilon))

        # the code below will be different from adversarial_test
        if self.denoised_method == 'threshold_method':
            x_test_denoised = ds.threshold_method(x_test_adversarial, thres)
        else:
            x_test_denoised = ds.threshold_method(x_test_adversarial, thres)
        accuracy, avg_confidence = self.evaluate(x_test_denoised, y_test_normal)

        print('* Adversarial Test with Denoising eps = %f, thres = %f\nAccuracy\tConfidence' % (epsilon, thres))
        print('%s\t\t%s' % (accuracy, avg_confidence))
        logging.info('%s\t%s\t\t%s\t%s' % ('denoised', accuracy, avg_confidence, epsilon))

    def flipping_test(self):
        # normal case
        test_size = 20000
        data, sess, x, y_, keep_prob = self.data, self.sess, self.x, self.y_, self.keep_prob
        x_test_normal = data.test.images[0:test_size]
        y_test_normal = data.test.labels[0:test_size]
        x_test_flipped = util.flip_black_white(x_test_normal)
        y_test_flipped = y_test_normal
        accuracy, avg_confidence = self.evaluate(x_test_flipped, y_test_flipped)
        print('* Flipping Test\nAccuracy\tConfidence')
        print('%s\t\t%s' % (accuracy, avg_confidence))
        logging.info('%s\t\t%s\t\t%s' % ('normal', accuracy, avg_confidence))

    def adversarialize(self, x_test_normal, y_test_normal, epsilon=0.1, adv_name=None):
        """
        adversarial case adding noise to the test data
        :param adv_name: the method to add noise
        :param epsilon: noise rate
        :param x_test_normal: normal test data
        :param y_test_normal: normal test label
        :return: x_test_adversarial, y_test_adversarial: adversarial test data & label
        """
        data, sess, x, y_, y, keep_prob = self.data, self.sess, self.x, self.y_, self.y, self.keep_prob
        J = self.cost_function  # cost function (cross entropy)

        # adversarial_method = getattr(ag, adversarial_method_name)
        if adv_name is None:
            adv_name = self.adv_method

        if adv_name == 'random_method':
            x_test_adversarial, noise = ag.random_method(J, x, y_, x_test_normal, y_test_normal, sess, keep_prob,
                                                         epsilon)
        else:  # 'fast_gradient_sign_method'
            x_test_adversarial, noise = ag.fast_gradient_sign_method(J, x, y_, x_test_normal, y_test_normal, sess,
                                                                     keep_prob, epsilon)
        # y_test_adversarial is the value calculated by network
        y_test_adversarial = sess.run(y, feed_dict={x: x_test_adversarial, keep_prob: 1.0})
        return x_test_adversarial, y_test_adversarial, noise

    def evaluate(self, x_test, y_test):
        # evaluate the model
        data, sess, x, y_, y, keep_prob = self.data, self.sess, self.x, self.y_, self.y, self.keep_prob
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        tmp = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy = sess.run(tmp, feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
        avg_confidence = sess.run(tf.reduce_mean(tf.reduce_max(y, axis=1)),
                                  feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
        return accuracy, avg_confidence

    def save_NvA_images(self, eps, image_num):
        assert self.cost_function is not None, "cost function can't be None, please train your network first"
        x = self.data.test.images[0:util.TEST_SIZE]
        y_ = self.data.test.labels[0:util.TEST_SIZE]
        y_norm = self.sess.run(self.y, feed_dict={self.x: x, self.keep_prob: 1.0})  # normal case
        x_adv, y_adv, noise = self.adversarialize(x, y_, eps)  # adversarial case
        x_denoised_flat = ds.threshold_method(x_adv, 0.5)
        y_denoised = self.sess.run(self.y, feed_dict={self.x: x_denoised_flat, self.keep_prob: 1.0})
        selected = list(random.sample(range(util.TEST_SIZE), image_num))

        for i in selected:
            x_2d = np.reshape(x[i], (28, 28))
            x_adv_2d = np.reshape(x_adv[i], (28, 28))
            noise_ad = np.reshape(noise[i], (28, 28))
            x_denoised_2d = np.reshape(x_denoised_flat[i], (28, 28))
            eps = noise_ad.max()

            # denoised = np.reshape(ds.threshold_method(x_adv[i],0.4), (28,28))
            x_adv_2d = (x_adv_2d + eps) / (1 + 2 * eps)
            noise_ad = (noise_ad + eps) * (0.5 / eps)

            label_correct = int(np.argmax(y_[i]))
            label_norm = int(np.argmax(y_norm[i]))
            conf_norm = np.amax(y_norm[i])
            label_adv = int(np.argmax(y_adv[i]))
            conf_adv = np.amax(y_adv[i])
            label_denoised = int(np.argmax(y_denoised[i]))
            conf_denoised = np.amax(y_denoised[i])

            self.save_image(x_2d, 'images\%d_%d_%.4f_%s' % (i, label_norm, conf_norm, 'nom'))
            self.save_image(x_adv_2d, 'images\%d_%d_%.4f_%s' % (i, label_adv, conf_adv, 'adv'))
            self.save_image(noise_ad, 'images\%d_%d_%s' % (i, label_correct, 'noise'))
            self.save_image(x_denoised_2d, 'images\%d_%d_%.4f_%s' % (i, label_denoised, conf_denoised, 'denoised'))

    @ staticmethod
    def save_image(image_matrix, label):
        toimage(image_matrix, cmin=0.0, cmax=1.0).save('%s.jpg' % label)

    def set_iter(self, iter):
        self.iter = iter


if __name__ == '__main__':
    # decide whether to start a new training or load the parameters from the result before
    new_training = False
    output_log = False
    output_img = False
    epsilon = 0.25

    # log file to save the network type, accuracy and average confidence
    logging.basicConfig(filename='normal_vs_adversarial.log', format='%(asctime)s %(message)s', level=logging.DEBUG)
    if not output_log:
        logging.disable(logging.INFO)

    training_algorithm = 'ReLU_Softmax_AdamOptimizer'
    # training_algorithm = 'Linear_Softmax_GradientDescentOptimizer'

    NvA = NormalVsAdversarial()
    if new_training:
        NvA.set_iter(1000)
        NvA.run_training(training_algorithm)
    else:
        NvA.restore_network(training_algorithm)

    # NvA.normal_test()
    NvA.adversarial_test_denoised(epsilon, 0.5)

    if output_img:
        NvA.save_NvA_images(epsilon, 10)

    NvA.sess.close()
