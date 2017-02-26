from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import toimage
from pylab import *

import tensorflow as tf
import adversarial_generator
import learning_strategy
import denoising_strategy

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
        x_test_normal = data.test.images
        y_test_normal = data.test.labels
        accuracy, avg_confidence = self.evaluate(x_test_normal, y_test_normal)
        print('* Normal Test\nAccuracy\tConfidence')
        print('%s\t\t%s' % (accuracy, avg_confidence))
        logging.info('%s\t\t%s\t\t%s' % ('normal', accuracy, avg_confidence))

    def adversarial_test(self, epsilon=0.1):
        # adversarial case
        test_size = 20000
        data, sess, x, y_, keep_prob = self.data, self.sess, self.x, self.y_, self.keep_prob
        x_test_normal = data.test.images
        y_test_normal = data.test.labels
        x_test_adversarial, y_test_adversarial, noise = self.adversarialize('fast_gradient_sign_method', x_test_normal,
                                                                            y_test_normal, epsilon)
        accuracy, avg_confidence = self.evaluate(x_test_adversarial, y_test_normal[0:test_size])
        print('* Adversarial Test\nAccuracy\tConfidence')
        print('%s\t\t%s' % (accuracy, avg_confidence))
        logging.info('%s\t%s\t\t%s\t%s' % ('adversarial', accuracy, avg_confidence, epsilon))

    def adversarial_test_denoised(self, epsilon=0.1, denoise_strategy_name='rof', para=0.5):
        test_size = 20000
        data, sess, x, y_, keep_prob = self.data, self.sess, self.x, self.y_, self.keep_prob
        x_test_normal = data.test.images
        y_test_normal = data.test.labels
        x_test_adversarial, y_test_adversarial, noise = self.adversarialize('fast_gradient_sign_method', x_test_normal,
                                                                            y_test_normal, epsilon)
        # the code below will be different from adversarial_test
        denoise_method = getattr(denoising_strategy, denoise_strategy_name)
        x_test_denoised = denoise_method(x_test_adversarial, para)
        accuracy, avg_confidence = self.evaluate(x_test_denoised, y_test_normal[0:test_size])
        print('* Adversarial Test with Denoising eps = %f, thres = %f\nAccuracy\tConfidence' % (epsilon, para))
        print('%s\t\t%s' % (accuracy, avg_confidence))
        logging.info('%s\t%s\t\t%s\t%s' % ('denoised', accuracy, avg_confidence, epsilon))

    def adversarialize(self, adversarial_method_name, x_test_normal, y_test_normal, epsilon=0.1):
        """
        adversarial case adding noise to the test data
        :param adversarial_method_name: the method to add noise
        :param epsilon: noise rate
        :param x_test_normal: normal test data
        :param y_test_normal: normal test label
        :return: x_test_adversarial, y_test_adversarial: adversarial test data & label
        """
        data, sess, x, y_, y, keep_prob = self.data, self.sess, self.x, self.y_, self.y, self.keep_prob
        adversarial_method = getattr(adversarial_generator, adversarial_method_name)
        J = self.cost_function  # cost function (cross entropy)
        x_test_adversarial, noise = adversarial_method(J, x, y_, x_test_normal, y_test_normal, sess, keep_prob, epsilon)
        # y_test_adversarial is the value calculated by network
        y_test_adversarial = sess.run(y, feed_dict={x: x_test_adversarial, keep_prob: 1.0})
        return x_test_adversarial, y_test_adversarial, noise

    def evaluate(self, x_test, y_test):
        # evaluate the model
        data, sess, x, y_, y, keep_prob = self.data, self.sess, self.x, self.y_, self.y, self.keep_prob
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        tmp = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy = sess.run(tmp, feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
        # accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
        avg_confidence = sess.run(tf.reduce_mean(tf.reduce_max(y, axis=1)),
                                  feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
        return accuracy, avg_confidence

    def save_NvA_images(self, data_set_name, image_num):
        assert self.cost_function is not None, "cost function can't be None, please train your network first"
        x = self.data.test.images
        y_ = self.data.test.labels
        y_norm = self.sess.run(self.y, feed_dict={self.x: x, self.keep_prob: 1.0})  # normal case
        x_adv, y_adv, noise = self.adversarialize('fast_gradient_sign_method', x, y_, 0.25)  # adversarial case
        # random_selected = random.sample(range(0, x.shape[0]), image_num)
        random_selected = random.sample(range(0, 10000), image_num)
        for i in random_selected:
            x_2d = np.reshape(x[i], (28, 28))
            x_adv_2d = np.reshape(x_adv[i], (28, 28))
            noise_ad = np.reshape(noise[i], (28, 28))
            # denoised = np.reshape(denoising_strategy.threshold_method(x_adv[i],0.4), (28,28))
            # x_adv_2d = (x_adv_2d + 0.25) / 1.5
            # x_adv_2d[x_adv_2d<0.4] = 0
            # noise_ad = (noise_ad + 0.25) * 2

            # kernel = np.ones((2, 1), np.uint8)
            # # erosion = cv2.erode(denoised,kernel,iterations = 1)
            # # opening = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
            # closing = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
            # plt.subplot(1, 3, 1)
            # plt.imshow(x_2d, 'gray')
            # plt.subplot(1, 3, 2)
            # plt.imshow(denoised, 'gray')
            # plt.subplot(1, 3, 3)
            # plt.imshow(closing, 'gray')
            # plt.show()

            label_correct = int(np.argmax(y_[i]))
            label_norm = int(np.argmax(y_norm[i]))
            conf_norm = np.amax(y_norm[i])
            label_adv = int(np.argmax(y_adv[i]))
            conf_adv = np.amax(y_adv[i])

            self.save_image(x_2d, 'images\%d_%d_%.4f_%s' % (i, label_norm, conf_norm, 'nom'))
            self.save_image(x_adv_2d, 'images\%d_%d_%.4f_%s' % (i, label_adv, conf_adv, 'adv'))
            self.save_image(noise_ad, 'images\%d_%d_%s' % (i, label_correct, 'noise'))
            # self.save_image(denoised, 'images\%d_%d_%s' % (i, label_correct, 'denoised'))

    def test_Images(self, image_num):
        x = self.data.test.images
        y_ = self.data.test.labels
        y_norm = self.sess.run(self.y, feed_dict={self.x: x, self.keep_prob: 1.0})  # normal case
        random_selected = random.sample(range(0, 10000), image_num)
        for i in random_selected:
            x_2d = np.reshape(x[i], (28, 28))
            figure()
            hist(x_2d.flatten(), 128)
            show()
            print(x_2d.min())

    def save_image(self, image_matrix, label):
        toimage(image_matrix, cmin=0.0, cmax=1.0).save('%s.jpg' % label)

    def set_iter(self, iter):
        self.iter = iter


if __name__ == '__main__':
    # decide whether to start a new training or load the parameters from the result before
    new_training = False
    output_log = False
    output_img = False

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

    NvA.normal_test()
    NvA.adversarial_test(0.25)
    NvA.adversarial_test_denoised(0.25, 'threshold_method', 0.5)

    # eps = 0.35
    # for thres in range(0,8):
    #     NvA.adversarial_test_denoised(eps,'threshold_method', 0.4 + thres*0.001)
    # NvA.test_Images(3)

    if output_img:
        NvA.save_NvA_images('MNIST_data', 3)

    NvA.sess.close()
