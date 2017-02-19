from tensorflow.examples.tutorials.mnist import input_data
from adversarial_generator import fast_gradient_sign_method
from learning_strategy import gradient_descent_optimizer, small_convolutional_neural_network

import tensorflow as tf

# initial dataset, tensorflow session, training data and training label
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

"""----- Gradient Descent Optimizer -----"""
print('===== Gradient Descent Optimizer =====')
# start training by gradient_descent_optimizer
iter = 1000  # iteration number for training
y = gradient_descent_optimizer(mnist, sess, x, y_, iter)
# initial test dataset
x_test_normal = mnist.test.images
y_test_normal = mnist.test.labels

# evaluate the model
# normal case
print('Accuracy of Normal test example')
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: x_test_normal, y_: y_test_normal}))

# adversarial case
# add "fast gradient sign method noise" to the test data
epsilon = 0.1  # noise rate, adjust this number to observe different results.
J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))  # cost function
x_test_adversarial = fast_gradient_sign_method(J, x, y_, x_test_normal, y_test_normal, sess, epsilon)
y_test_adversarial = y_test_normal  # y_test_adversarial_ is same as the normal one
print('Accuracy of Adversarial test example')
print(accuracy.eval(feed_dict={x: x_test_adversarial, y_: y_test_adversarial}))

"""----- Small convolutional neural network -----"""
print('===== Small Convolutional Neural Network =====')
keep_prob = tf.placeholder(tf.float32)  # dropout probability to avoid overfitting
iter = 20000  # iteration number for training
y_conv = small_convolutional_neural_network(mnist, sess, x, y_, keep_prob, iter)

# evaluate the model
# normal case
print('Accuracy of Normal test example')
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: x_test_normal, y_: y_test_normal, keep_prob: 1.0}))

# adversarial case
# add "fast gradient sign method noise" to the test data
epsilon = 0.25  # noise rate, adjust this number to observe different results.
J = cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))  # cost function
# adversarial case
x_test_adversarial = fast_gradient_sign_method(J, x, y_conv, x_test_normal, y_test_normal, sess, epsilon)
y_test_adversarial = y_test_normal  # y_test_adversarial_ is same as the normal one

print('Accuracy of Adversarial test example')
print(accuracy.eval(feed_dict={x: x_test_adversarial, y_: y_test_adversarial, keep_prob: 1.0}))
