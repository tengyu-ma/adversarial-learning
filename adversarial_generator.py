import tensorflow as tf
import numpy as np

def fast_gradient_sign_method(J, x, y_, x_test, y_test, sess, keep_prob, drop_out=False, epsilon=0.1):
    """
    A fast gradient sign method to generate adversarial example
    :param J: cost function
    :param x: test data placeholder
    :param y_: true y_ label placeholder
    :param x_test: test data
    :param y_test: test label
    :param sess: the tensorflow session
    :param drop_out: dropout probability
    :param epsilon: noise rate
    :return: x_adversarial, training data adding fast gradient sign noise
        x_adversarial = x + sign(▽_xJ(theta,x,y))
    """

    nabla_J = tf.gradients(J, x)  # apply nabla operator to calculate the gradient
    sign_nabla_J = tf.sign(nabla_J)  # calculate the sign of the gradient of cost function
    eta = tf.multiply(sign_nabla_J, epsilon)  # multiply epsilon the sign of the gradient of cost function
    x_test_adversarial = []

    test_size = x_test.size // 784
    for i in range(int(test_size // 1000)):
        x_temp = x_test[i * 1000:(i + 1) * 1000]
        y_temp = y_test[i * 1000:(i + 1) * 1000]
        if drop_out:
            eta_result = sess.run(eta, feed_dict={x: x_temp, y_: y_temp, keep_prob: 1.0})
        else:
            eta_result = sess.run(eta, feed_dict={x: x_temp, y_: y_temp})

        eta_flatten = eta_result[0]  # since we only have 1 row, we need to flat the eta. e.g. [[[a1,a2,..,an]]] to [[a1,a2,..,an]]
        temp = sess.run(tf.add(x_temp, eta_flatten))  # add noise to test data
        if x_test_adversarial == []:
            x_test_adversarial = temp
        else:
            x_test_adversarial = np.vstack((x_test_adversarial,temp))

    return x_test_adversarial
