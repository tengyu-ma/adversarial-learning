import tensorflow as tf
import numpy as np
import random
import os
import denoise
from scipy.misc import toimage
from utils import util


def fgsm_mnist(J, x, y_, x_test, y_test, sess, keep_prob, epsilon=0.1):
    """
    A fast gradient sign method to generate adversarial example
    
    Parameters
    ----------
    J : 
        cost function
    x : 
        test data placeholder
    y_ : 
        true y_ label placeholder
    x_test : 
        test data
    y_test : 
        test label
    sess : 
        the tensorflow session
    keep_prob :
        dropout probability
    epsilon :
        noise rate

    Returns
    -------
    x_test_adversarial : 
        training data adding fast gradient sign noise; x_adversarial = x + sign(▽_xJ(theta,x,y))
    noise : 
        the noise added to the normal data; noise = sign(▽_xJ(theta,x,y))
    """

    nabla_J = tf.gradients(J, x)  # apply nabla operator to calculate the gradient
    sign_nabla_J = tf.sign(nabla_J)  # calculate the sign of the gradient of cost function
    eta = tf.multiply(sign_nabla_J, epsilon)  # multiply epsilon the sign of the gradient of cost function
    eta_flatten = tf.reshape(eta, [-1, 784])
    x_test_adversarial = np.array([])
    noise = np.array([])

    test_size = x_test.shape[0]
    test_batch = 1000
    test_loop = test_size // test_batch
    for i in range(test_loop):
        x_temp = x_test[i * test_batch:(i + 1) * test_batch]
        y_temp = y_test[i * test_batch:(i + 1) * test_batch]
        eta_result = sess.run(eta_flatten, feed_dict={x: x_temp, y_: y_temp, keep_prob: 1.0})
        # add noise to test data
        temp = sess.run(tf.add(x_temp, eta_result), feed_dict={x: x_temp, y_: y_temp, keep_prob: 1.0})
        # temp = sess.run(tf.subtract(x_temp, eta_result), feed_dict={x: x_temp, y_: y_temp, keep_prob: 1.0})
        if not x_test_adversarial.size:
            x_test_adversarial = temp
            noise = eta_result
        else:
            x_test_adversarial = np.vstack((x_test_adversarial, temp))
            noise = np.vstack((noise, eta_result))

    return x_test_adversarial, noise


def fgsm_imagenet(sess, J, x, y_, rn_x, image_data, image_label, epsilon=2):
    """
    A fast gradient sign method to generate adversarial example

    Parameters
    ----------
    sess : Session
        default tensorflow session
    J : tensor
        cost function
    x : placeholder
        placeholder for the input image
    y_ : placeholder
        placeholder for the correct one-hot label
    rn_x : tensor
        tensorflow tensor for the resized and normalized input image
    image_data : ndarray
        resized and normalized input data
    image_label : ndarray
        one-hot correct label
    epsilon :
        noise rate

    Returns
    -------
    x_test_adversarial : 
        training data adding fast gradient sign noise; x_adversarial = x + sign(▽_xJ(theta,x,y))
    noise : 
        the noise added to the normal data; noise = sign(▽_xJ(theta,x,y))
    """
    nabla_J = tf.gradients(J, rn_x)  # apply nabla operator to calculate the gradient
    sign_nabla_J = tf.sign(nabla_J)  # calculate the sign of the gradient of cost function
    eta = tf.multiply(sign_nabla_J, epsilon)  # multiply epsilon the sign of the gradient of cost function

    noise = sess.run(eta, {x: image_data, y_: image_label})
    x_adv = sess.run(tf.add(rn_x, tf.squeeze(noise)), {x: image_data, y_: image_label})  # add noise to test data

    return x_adv, noise


def fgsm_imagenet_batch(sess, x_adv_tensor, eta, J, x, y_, rn_x, image_data, image_label, epsilon=2):
    """
    A fast gradient sign method to generate adversarial example

    Parameters
    ----------
    sess : Session
        default tensorflow session
    x_adv_tensor : tensor
        tensor for adversarial image
    eta : tensor
        tensor for adversarial noise
    J : tensor
        cost function
    x : placeholder
        placeholder for the input image
    y_ : placeholder
        placeholder for the correct one-hot label
    rn_x : tensor
        tensorflow tensor for the resized and normalized input image
    image_data : ndarray
        resized and normalized input data
    image_label : ndarray
        one-hot correct label
    epsilon :
        noise rate

    Returns
    -------
    x_test_adversarial : 
        training data adding fast gradient sign noise; x_adversarial = x + sign(▽_xJ(theta,x,y))
    noise : 
        the noise added to the normal data; noise = sign(▽_xJ(theta,x,y))
    """
    # nabla_J = tf.gradients(J, rn_x)  # apply nabla operator to calculate the gradient
    # sign_nabla_J = tf.sign(nabla_J)  # calculate the sign of the gradient of cost function
    # eta = tf.multiply(sign_nabla_J, epsilon)  # multiply epsilon the sign of the gradient of cost function

    noise = sess.run(eta, {rn_x: image_data, y_: image_label})
    # noise = None
    x_adv = sess.run(x_adv_tensor, {rn_x: image_data, y_: image_label})  # add noise to test data
    # x_max = np.amax(x_adv)
    x_adv = x_adv / np.amax(x_adv)
    # save_image(image_data, 'fgsm_imagedata')
    # save_image(noise, 'fgsm_noise')
    # save_image(x_adv, 'fgsm_adv_imagedata')
    # denoise.random_cover_imagenet(x_adv)

    return x_adv, noise


def random_imagenet_batch(J, x, y_, rn_x, image_data, image_label, epsilon=2):
    """
    A random sign method to generate adversarial example

    Parameters
    ----------
    J : tensor
        cost function
    x : placeholder
        placeholder for the input image
    y_ : placeholder
        placeholder for the correct one-hot label
    rn_x : tensor
        tensorflow tensor for the resized and normalized input image
    image_data : ndarray
        resized and normalized input data
    image_label : ndarray
        one-hot correct label
    epsilon :
        noise rate

    Returns
    -------
    x_test_adversarial : 
        training data adding fast gradient sign noise; x_adversarial = x + sign(▽_xJ(theta,x,y))
    noise : 
        the noise added to the normal data; noise = sign(▽_xJ(theta,x,y))
    """
    # nabla_J = tf.gradients(J, rn_x)  # apply nabla operator to calculate the gradient
    # sign_nabla_J = tf.sign(nabla_J)  # calculate the sign of the gradient of cost function
    # eta = tf.multiply(sign_nabla_J, epsilon)  # multiply epsilon the sign of the gradient of cost function

    random_sign = np.random.randint(0, 2, size=(299, 299, 3))
    random_sign[random_sign == 0] = -1
    noise = random_sign * epsilon
    x_adv = image_data + noise

    return x_adv, noise


def fgsm_imagenet_batch_init(J, rn_x, epsilon=2):
    """
    A fast gradient sign method to generate adversarial example

    Parameters
    ----------
    J : tensor
        cost function
    rn_x : tensor
        tensorflow tensor for the resized and normalized input image
    epsilon :
        noise rate

    Returns
    -------
    x_adv_tensor : 
        tensor for adversarial image
    eta : 
        the noise tensor added to the normal data; noise = sign(▽_xJ(theta,x,y))
    """
    nabla_J = tf.gradients(J, rn_x)  # apply nabla operator to calculate the gradient
    sign_nabla_J = tf.sign(nabla_J)  # calculate the sign of the gradient of cost function
    eta = tf.multiply(sign_nabla_J, epsilon)  # multiply epsilon the sign of the gradient of cost function
    # x_adv_tensor = tf.add(rn_x, tf.squeeze(eta))
    x_adv_tensor = tf.subtract(rn_x, tf.squeeze(eta))
    # noise = sess.run(eta, {rn_x: image_data, y_: image_label})
    # noise = None
    # x_adv = sess.run(tf.add(rn_x, tf.squeeze(eta)), {rn_x: image_data, y_: image_label})  # add noise to test data

    # eta = nabla_J
    return x_adv_tensor, eta


def random_method(J, x, y_, x_test, y_test, sess, keep_prob, epsilon=0.1):
    x_test_adversarial = np.array([])
    noise = np.array([])
    eta_flatten = np.array([])

    test_size = x_test.shape[0]
    test_batch = 1000
    test_loop = test_size // test_batch

    for i in range(test_loop):
        x_temp = x_test[i * test_batch:(i + 1) * test_batch]
        y_temp = y_test[i * test_batch:(i + 1) * test_batch]
        # since we only have 1 row, we need to flat the eta. e.g. [[[a1,a2,..,an]]] to [[a1,a2,..,an]]

        eta_flatten = np.array([random.choice([-1, 1]) * epsilon for i in range(784)])
        for _ in range(1, test_batch):
            tmp1 = np.array([random.choice([-1, 1]) * epsilon for i in range(784)])
            eta_flatten = np.vstack((eta_flatten, tmp1))

        temp = sess.run(tf.add(x_temp, eta_flatten))  # add noise to test data
        if not x_test_adversarial.size:
            x_test_adversarial = temp
            noise = eta_flatten
        else:
            x_test_adversarial = np.append(x_test_adversarial, temp, axis=0)
            noise = np.vstack((noise, eta_flatten))

    return x_test_adversarial, noise


def save_image(image_matrix, label):
    image_matrix = np.squeeze(image_matrix)
    save_path = os.path.join(util.ImageNet_DATA_DIR, "%s.jpg" % label)
    toimage(image_matrix).save(save_path)
