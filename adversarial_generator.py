import tensorflow as tf
import numpy as np
import random


def fast_gradient_sign_method(J, x, y_, x_test, y_test, sess, keep_prob, epsilon=0.1):
    """
    A fast gradient sign method to generate adversarial example
    :param J: cost function
    :param x: test data placeholder
    :param y_: true y_ label placeholder
    :param x_test: test data
    :param y_test: test label
    :param sess: the tensorflow session
    :param keep_prob: dropout probability
    :param epsilon: noise rate
    :return: x_adversarial, training data adding fast gradient sign noise
        x_adversarial = x + sign(▽_xJ(theta,x,y))
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
        temp = sess.run(tf.add(x_temp, eta_result))  # add noise to test data
        if not x_test_adversarial.size:
            x_test_adversarial = temp
            noise = eta_result
        else:
            x_test_adversarial = np.vstack((x_test_adversarial, temp))
            noise = np.vstack((noise, eta_result))

    return x_test_adversarial, noise


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
        for _ in range(1,test_batch):
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
