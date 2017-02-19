import tensorflow as tf


def Linear_Softmax_GradientDescentOptimizer(data, sess, x, y_, keep_prob, iter=1000):
    """
    a gradient descent optimizer
    :param data: dataset
    :param sess: tensorflow session
    :param x: tensorflow placeholder for training data
    :param y_: tensorflow placeholder for training label
    :param keep_prob: drop out probability
    :param iter: iteration time for training
    :return: y: target function / predicted value
    """
    # target function
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    sess.run(tf.global_variables_initializer())
    y = tf.matmul(x, W) + b

    # loss function or cost function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    # steepest gradient descent
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    for _ in range(iter):
        batch = data.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    return tf.nn.softmax(y), cross_entropy


def ReLU_Softmax_AdamOptimizer(data, sess, x, y_, keep_prob, iter=20000):
    """
    a small convolutional neural network
    :param data: dataset
    :param sess: tensorflow session
    :param x: tensorflow placeholder for training data
    :param y_: tensorflow placeholder for training label
    :param keep_prob: probability of dropout
    :param iter: iteration time for training
    :return: y_conv: target function / predicted value
    """
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # First Convolutional Layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolutional Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout in order to avoid overfitting
    # keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # similar cost function as Gradient Descent Optimizer which is cross_entropy
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # evaluate the model for each training step
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())
    # start training
    for i in range(iter):
        batch = data.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    return tf.nn.softmax(y_conv), cross_entropy
