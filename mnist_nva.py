from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import toimage
from pylab import *

import tensorflow as tf
import numpy as np
import logging
import random
import os
import cv2

from models.mnist_model import ReLU_Softmax_AdamOptimizer as mnist_DNNs
from utils import util
import adversarize
import denoise


class MnistNvA:
    """A class of training, evaluation, adversarizing, denoising for MNIST dataset.
    
    The DNNs used in MNIST is the exact same as the tutorial in Tensorflow:
    https://www.tensorflow.org/get_started/mnist/pros

    Attributes
    ----------
    data : dataset, 'MNIST_data'
        the MNIST dataset
    sess : session
        the TensorFlow session
    x : placeholder
        the TensorFlow placeholder for the input of MNIST dataset
    y_ : placeholder
        the TensorFlow placeholder for the correct label of the corresponding input data
    y : tensor
        the TensorFlow tensor representing the predicted value inferred by the DNNs
    J : tensor
        the TensorFlow tensor representing the cost function given DNNs
    keep_prob : placeholder
        the TensorFlow placeholder for the keep probability of drop out
    adv_method : str, 'fgsm'
        the adversarial method used to generate adversarial examples
    denoise_method: str, 'threshold_method'
        the denoise method used to remove adversarial noise

    """

    def __init__(self, adv_method='fgsm', denoise_method='threshold_method'):
        self.data = input_data.read_data_sets('data/mnist/nom', one_hot=True)
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        self.y = None
        self.J = None
        self.keep_prob = tf.placeholder(tf.float32)  # drop out probability of the network
        self.adv_method = adv_method
        self.denoise_method = denoise_method

    def training(self, train_iter=1000, restore=False):
        """Run training for the default DNNs.
        
        Parameters
        ----------
        train_iter : int, 1000
            The number of iteration used to train the DNNs
        restore : boolean, False
            The flag indicates training from scratch or restored from a checkpoint 
            
        """
        self.y, self.J = mnist_DNNs(self.data,
                                                self.sess,
                                                self.x,
                                                self.y_,
                                                self.keep_prob,
                                                train_iter,
                                                restore)

        # save the new trained model
        if not restore:
            save_path = ".%scheckpoint%s%s.ckpt" % (os.sep, os.sep, "mnist_DNNs")
            saver = tf.train.Saver()
            save_path = saver.save(self.sess, save_path)
            print("[+] Model saved in file: %s" % save_path)

    def nom_test(self):
        """Evaluation on the normal examples of the testing dataset
        """
        # get normal testing data
        x_test_normal = self.data.test.images[0:util.TEST_SIZE]
        y_test_normal = self.data.test.labels[0:util.TEST_SIZE]

        accuracy, avg_confidence = self.evaluate(x_test_normal, y_test_normal)

        print('* Normal Test\nAccuracy\tConfidence')
        print('%s\t\t%s' % (accuracy, avg_confidence))

    def adv_test(self, adv_name='fgsm', epsilon=0.1):
        """Evaluation on the adversarial examples of the testing dataset
        
        Parameters
        ----------
        adv_name : str, 'fgsm'
            the method used to generate adversarial examples
        epsilon : float, 0.1
            the noise rate

        """
        # get normal testing data
        x_test_normal = self.data.test.images[0:util.TEST_SIZE]
        y_test_normal = self.data.test.labels[0:util.TEST_SIZE]

        # generate adversarial testing data
        x_test_adversarial, y_test_adversarial, noise = self.adversarize(x_test_normal, y_test_normal,
                                                                         adv_name, epsilon)

        accuracy, avg_confidence = self.evaluate(x_test_adversarial, y_test_normal)

        print('* Adversarial Test\nAccuracy\tConfidence')
        print('%s\t\t%s' % (accuracy, avg_confidence))

    def adversarize(self, x_test_normal, y_test_normal, adv_name='fgsm', epsilon=0.1):
        """The function used to generate adversarial examples
        
        Parameters
        ----------
        x_test_normal : ndarray
            The normal testing dataset
        y_test_normal : ndarray
            The corresponding correct labels for the testing dataset
        adv_name : str, 'fgsm'
            The method used to generate adversarial examples
        epsilon : float, 0.1
            The adversarial rate

        Returns
        -------
        x_test_adversarial : ndarray
            The adversarial testing dataset
        y_test_adversarial : ndarray
            The corresponding correct labels for the testing dataset
        noise : ndarray
            The noise added to the normal dataset

        """

        if adv_name == 'fgsm':
            x_test_adversarial, noise = adversarize.fgsm_mnist(self.J,
                                                               self.x, self.y_,
                                                               x_test_normal, y_test_normal,
                                                               self.sess,
                                                               self.keep_prob,
                                                               epsilon)
        elif adv_name == 'random_method':
            x_test_adversarial, noise = adversarize.random_method(self.J,
                                                                  self.x, self.y_,
                                                                  x_test_normal, y_test_normal,
                                                                  self.sess,
                                                                  self.keep_prob,
                                                                  epsilon)
        else:
            raise Exception("Unknown adversarial method: %s \n" % adv_name)

        y_test_adversarial = self.sess.run(self.y,
                                           feed_dict={self.x: x_test_adversarial,
                                                      self.keep_prob: 1.0})

        return x_test_adversarial, y_test_adversarial, noise

    def denoise_test(self, epsilon=0.1, thres=0.5, adv_name='fgsm', denoise_name=None):
        """Evaluation on the denoise examples of the testing dataset
        
        Parameters
        ----------
        epsilon : float, 0.1
            the adversarial rate
        thres : float, 0.5 
            threshold for the threshold method
        adv_name : str, 'fgsm'
            The method used to generate adversarial examples
        denoise_name : str
            the denoise method used to remove adversarial noise

        """
        assert denoise_name is not None, "Denoise name can not be none!"

        # get normal testing data
        x_test_normal = self.data.test.images[0:util.TEST_SIZE]
        y_test_normal = self.data.test.labels[0:util.TEST_SIZE]

        # generate adversarial testing data
        x_test_adversarial, y_test_adversarial, noise = self.adversarize(x_test_normal, y_test_normal,
                                                                         adv_name, epsilon)

        # Evaluate accuracy and confidence on the adversarial testing data
        accuracy, avg_confidence = self.evaluate(x_test_adversarial, y_test_normal)

        print('* Adversarial Test, eps = %f\nAccuracy\tConfidence' % epsilon)
        print('%s\t\t%s' % (accuracy, avg_confidence))

        if denoise_name == 'threshold_method':
            x_test_denoised = denoise.threshold_method(x_test_adversarial, thres)
        elif denoise_name == 'test_method':
            x_test_denoised = denoise.test_method(x_test_adversarial, thres)
        else:
            raise Exception("Unknown denoise method: %s \n" % denoise_name)

        # Evaluate accuracy and confidence on the denoise testing data
        accuracy, avg_confidence = self.evaluate(x_test_denoised, y_test_normal)

        print('* Adversarial Test with Denoising, eps = %f, thres = %f\nAccuracy\tConfidence' % (epsilon, thres))
        print('%s\t\t%s' % (accuracy, avg_confidence))

    def evaluate(self, x_test, y_test):
        """Evaluation accuracy and confidence given the testing data.
        
        Parameters
        ----------
        x_test : ndarray
            The dataset of testing data
        y_test : ndarray
            The corresponding correct label for the testing data

        Returns
        -------
        accuracy : float
            The average classification accuracy across the whole testing dataset 
        avg_confidence : float
            The average classification confidence across the whole testing dataset 
                        
        """
        correct_prediction = tf.equal(tf.argmax(self.y, 1),
                                      tf.argmax(self.y_, 1))
        tmp = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy = self.sess.run(tmp,
                                 feed_dict={self.x: x_test,
                                            self.y_: y_test,
                                            self.keep_prob: 1.0})
        avg_confidence = self.sess.run(tf.reduce_mean(tf.reduce_max(self.y, axis=1)),
                                       feed_dict={self.x: x_test,
                                                  self.y_: y_test,
                                                  self.keep_prob: 1.0})
        return accuracy, avg_confidence

    def save_nva_images(self, eps, thres, image_num, adv_name='fgsm', denoise_name='threshold_method'):
        """Save all four images for our testing results:
        1) normal images 2) adversarial images 3) noise images 4) denoise images
        
        Parameters
        ----------
        eps : float
            adversarial noise rate
        thres : float
            threshold value for threshold denoise method
        image_num : int
            the number of saved images which are sampled from the whole testing dataset
        adv_name : str, 'fgsm'
            the method used to generate adversarial examples
        denoise_name : str, 'threshold_method'
            the method used to remove adversarial noise

        """
        assert self.J is not None, "cost function can't be None, please train your network first"

        # get testing dataset
        x_test = self.data.test.images[0:util.TEST_SIZE]
        y_test = self.data.test.labels[0:util.TEST_SIZE]

        # inference for the predicted label
        y_nom = self.sess.run(self.y, feed_dict={self.x: x_test, self.keep_prob: 1.0})

        # generate adversarial examples
        if adv_name == 'fgsm':
            x_adv, y_adv, noise = self.adversarize(x_test, y_test, adv_name, eps)
        else:
            raise Exception("Unknown adversarial method: %s \n" % adv_name)

        # denoise adversarial examples
        if denoise_name == 'threshold_method':
            x_denoise_flat = denoise.threshold_method(x_adv, thres)
        else:
            raise Exception("Unknown denoise method: %s \n" % denoise_name)
        y_denoise = self.sess.run(self.y, feed_dict={self.x: x_denoise_flat, self.keep_prob: 1.0})

        # random sample the required number of images to save
        selected = list(random.sample(range(util.TEST_SIZE), image_num))
        for i in selected:
            x_2d = np.reshape(x_test[i], (28, 28))
            x_adv_2d = np.reshape(x_adv[i], (28, 28))
            noise_ad = np.reshape(noise[i], (28, 28))
            x_denoised_2d = np.reshape(x_denoise_flat[i], (28, 28))
            eps = noise_ad.max()

            # denoised = np.reshape(ds.threshold_method(x_adv[i],0.4), (28,28))
            x_adv_2d = (x_adv_2d + eps) / (1 + 2 * eps)
            noise_ad = (noise_ad + eps) * (0.5 / eps)

            label_correct = int(np.argmax(y_test[i]))
            label_norm = int(np.argmax(y_nom[i]))
            conf_norm = np.amax(y_nom[i])
            label_adv = int(np.argmax(y_adv[i]))
            conf_adv = np.amax(y_adv[i])
            label_denoise = int(np.argmax(y_denoise[i]))
            conf_denoise = np.amax(y_denoise[i])

            # save images
            self.save_image(x_2d, 'data/mnist/adv/%d_%d_%.4f_%s' % (i, label_norm, conf_norm, 'nom'))
            self.save_image(x_adv_2d, 'data/mnist/adv/%d_%d_%.4f_%s' % (i, label_adv, conf_adv, 'adv'))
            self.save_image(noise_ad, 'data/mnist/adv/%d_%d_%s' % (i, label_correct, 'noise'))
            self.save_image(x_denoised_2d, 'data/mnist/adv/%d_%d_%.4f_%s' % (i, label_denoise, conf_denoise, 'denoised'))

    @staticmethod
    def save_image(image_matrix, label):
        toimage(image_matrix, cmin=0.0, cmax=1.0).save('%s.jpg' % label)

    # def set_iter(self, train_iter):
    #     self.train_iter = train_iter


if __name__ == '__main__':
    restore = False
    output_img = False
    epsilon = 0.25
    thres = 0.5
    train_iter = 1000

    mnist_nva = MnistNvA()

    mnist_nva.training(train_iter, restore)

    mnist_nva.nom_test()
    mnist_nva.adv_test('fgsm', epsilon)
    mnist_nva.denoise_test(epsilon, thres, 'fgsm', 'threshold_method')

    if output_img:
        mnist_nva.save_nva_images(epsilon, thres, 10, 'fgsm', 'threshold_method')

    mnist_nva.sess.close()
