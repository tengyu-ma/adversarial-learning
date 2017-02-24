from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import toimage

import tensorflow as tf
import adversarial_generator
import learning_strategy

import numpy as np
import logging
import random


class NormalVsAdversarial:
    def __init__(self, data_set_name='MNIST_data', iter=1000):
        self.data = input_data.read_data_sets(data_set_name, one_hot=True)  # dataset
        self.sess = tf.InteractiveSession()  # tensorflow session
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
        self.y, self.cost_function = learning_model(data, self.sess, x, y_, keep_prob, iter)

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
        data, sess, x, y_, keep_prob = self.data, self.sess, self.x, self.y_, self.keep_prob
        x_test_normal = data.test.images
        y_test_normal = data.test.labels
        x_test_adversarial, y_test_adversarial, noise = self.adversarialize('fast_gradient_sign_method', x_test_normal,
                                                                            y_test_normal, epsilon)
        accuracy, avg_confidence = self.evaluate(x_test_adversarial, y_test_normal)
        print('* Adversarial Test\nAccuracy\tConfidence')
        print('%s\t\t%s' % (accuracy, avg_confidence))
        logging.info('%s\t%s\t\t%s\t%s' % ('adversarial', accuracy, avg_confidence, epsilon))

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
        y_test_adversarial = sess.run(y, feed_dict={x: x_test_adversarial,
                                                    keep_prob: 1.0})  # y_test_adversarial is the value calculated by network
        return x_test_adversarial, y_test_adversarial, noise

    def evaluate(self, x_test, y_test):
        # evaluate the model
        data, sess, x, y_, y, keep_prob = self.data, self.sess, self.x, self.y_, self.y, self.keep_prob
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).eval(
            feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
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
        random_selected = random.sample(range(0, x.shape[0]), image_num)
        for i in random_selected:
            x_2d = np.reshape(x[i], (28, 28))
            x_adv_2d = np.reshape(x_adv[i], (28, 28))
            noise_ad = np.reshape(noise[i], (28, 28))

            label_correct = int(np.argmax(y_[i]))
            label_norm = int(np.argmax(y_norm[i]))
            conf_norm = np.amax(y_norm[i])
            label_adv = int(np.argmax(y_adv[i]))
            conf_adv = np.amax(y_adv[i])

            self.save_image(x_2d, 'images\%d_%d_%.4f_%s' % (i, label_norm, conf_norm, 'nom'))
            self.save_image(x_adv_2d, 'images\%d_%d_%.4f_%s' % (i, label_adv, conf_adv, 'adv'))
            self.save_image(noise_ad, 'images\%d_%d_%s' % (i, label_correct, 'noise'))

    def save_image(self, image_matrix, label):
        toimage(image_matrix, cmin=0.0, cmax=1.0).save('%s.jpg' % label)

    def set_iter(self, iter):
        self.iter = iter


if __name__ == '__main__':
    # log file to save the network type, accuracy and average confidence
    logging.basicConfig(filename='normal_vs_adversarial.log', format='%(asctime)s %(message)s', level=logging.DEBUG)

    NvA = NormalVsAdversarial()
    # NvA.run_training('Linear_Softmax_GradientDescentOptimizer')
    # NvA.normal_test()
    # NvA.adversarial_test()
    #
    NvA.set_iter(20000)
    NvA.run_training('ReLU_Softmax_AdamOptimizer')
    NvA.normal_test()
    NvA.adversarial_test(0.25)

    NvA.save_NvA_images('MNIST_data', 100)
