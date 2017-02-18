from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())
W_init = sess.run(W)
b_init = sess.run(b)
print('===== initial parameters =====')
print('W_init: ', W_init, 'b_init: ', b_init)
# target function
y = tf.matmul(x, W) + b
# loss function or cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# steepest gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print('===== trained parameters =====')
# add "fast gradient sign method noise" to the test images
W_trained = sess.run(W)  # W value after trained
b_trained = sess.run(b)  # b value after trained
print('W_trained: ', W_trained, 'b_trained: ', b_trained)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# test set
print('===== Accuracy of Normal test example =====')
x_test_normal = mnist.test.images
y_test_normal = mnist.test.labels
print(accuracy.eval(feed_dict={x: x_test_normal, y_: y_test_normal}))


y = tf.matmul(x, W) + b  # W and b has already trained to fit the training data at this time
epsilon = 0.1  # noise rate, adjust this number to observe different results. 0.1 will cause

J = cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))  # cost function
nabla_J = tf.gradients(J, x)  # apply nabla operator to calculate the gradient
sign_nabla_J = tf.sign(nabla_J)  # calculate the sign of the gradient of cost function
eta = tf.mul(sign_nabla_J, epsilon)  # multiply epsilon the sign of the gradient of cost function
eta_result = sess.run(eta, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
eta_flatten = eta_result[0]  # since we only have 1 row, we need to flat the eta. e.g. [[[a1,a2,..,an]]] to [[a1,a2,..,an]]

x_test_adversarial = sess.run(tf.add(x_test_normal, eta_flatten))  # add noise to test data
y_test_adversarial = y_test_normal  # y_test_adversarial_ is same as the normal one

print('===== Accuracy of Adversarial test example =====')
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: x_test_adversarial, y_: y_test_adversarial}))

print('Our classifier is the graident_optimizer, hence, we have a worse result than the softmax classifier \
    According to paper: for softmax classifier using epsilon = 0.25, the correct rate is only 0.001. The \
    same epsilon will cause ours correct rate to be 0! If epsilon = 0.1, our correct rate is around 0.08')

'''
In the "EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES" paper:
We find that this method reliably causes a wide variety of models to misclassify their input. See
Fig. 1 for a demonstration on ImageNet. We find that using  epsilon = 0.25, we cause a shallow softmax
classifier to have an error rate of 99.9% with an average confidence of 79.3% on the MNIST (?) test
set1. In the same setting, a maxout network misclassifies 89.4% of our adversarial examples with
an average confidence of 97.6%. Similarly, using epsilon = 0.1, we obtain an error rate of 87.15% and
an average probability of 96.6% assigned to the incorrect labels when using a convolutional maxout
network on a preprocessed version of the CIFAR-10 (Krizhevsky & Hinton, 2009) test set2.
'''

