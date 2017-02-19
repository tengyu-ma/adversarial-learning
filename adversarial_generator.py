import tensorflow as tf


def fast_gradient_sign_method(J, x, y_, x_test, y_test, sess, epsilon=0.1):
    """
    A fast gradient sign method to generate adversarial example
    :param J: cost function
    :param x: test data placeholder
    :param y_: true y_ label placeholder
    :param x_test: test data
    :param y_test: test label
    :param sess: the tensorflow session
    :param epsilon: noise rate
    :return: x_adversarial, training data adding fast gradient sign noise
        x_adversarial = x + sign(▽_xJ(theta,x,y))
    """

    nabla_J = tf.gradients(J, x)  # apply nabla operator to calculate the gradient
    sign_nabla_J = tf.sign(nabla_J)  # calculate the sign of the gradient of cost function
    eta = tf.mul(sign_nabla_J, epsilon)  # multiply epsilon the sign of the gradient of cost function
    eta_result = sess.run(eta, feed_dict={x: x_test, y_: y_test})
    eta_flatten = eta_result[0]  # since we only have 1 row, we need to flat the eta. e.g. [[[a1,a2,..,an]]] to [[a1,a2,..,an]]

    x_test_adversarial = sess.run(tf.add(x_test, eta_flatten))  # add noise to test data
    return x_test_adversarial
