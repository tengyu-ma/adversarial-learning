from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from settings import *

import math
import time
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cifar10
import cv2
from copy import deepcopy
from PIL import Image


def generate_images_size24():
    with tf.variable_scope('network1') as scope:
        batch_size = FLAGS.batch_size
        FLAGS.image_size = 32
        FLAGS.batch_size = 1
        images, labels, images_org = cifar10.inputs(eval_data=FLAGS.eval_data)
        logits = cifar10.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                     start=True))

                step = 0
                true_count = 0
                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
                total_sample_count = num_iter * FLAGS.batch_size
                while step < num_iter and not coord.should_stop():
                    predictions, images_array_org, images_array, labels_array = sess.run(
                        [top_k_op, images_org, images, labels])
                    image_matrix_org = np.array([images_array_org[:, :, 0], images_array_org[:, :, 1],
                                                 images_array_org[:, :, 2]])
                    image_list_org = np.array(image_matrix_org.flatten())
                    image_list_org = image_list_org.astype('uint8')
                    image_list_org = list(image_list_org)
                    label_list_org = list(map(int, list(labels_array)))
                    data_list_org = np.array(label_list_org + image_list_org)
                    data_list_org = list(data_list_org)
                    data_bytes_org = bytes(data_list_org)

                    if step == 0:
                        file_org = data_bytes_org
                    else:
                        file_org += data_bytes_org

                    if step % 200 == 0:
                        print("Step: %d" % step)
                    true_count += np.sum(predictions)
                    step += 1

                file_to_write_org = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin/test_batch_size24.bin')
                with open(file_to_write_org, 'wb') as f:
                    f.write(file_org)

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

        FLAGS.batch_size = batch_size


def generate_images_with_noise():
    with tf.variable_scope('network1') as scope:
        batch_size = FLAGS.batch_size
        FLAGS.image_size = 24
        FLAGS.batch_size = 1
        images, labels, images_org = cifar10.inputs(eval_data=FLAGS.eval_data)
        logits = cifar10.inference(images)

        loss = cifar10.loss(logits, labels)
        nabla_J = tf.gradients(loss, images)  # apply nabla operator to calculate the gradient
        sign_nabla_J = tf.sign(nabla_J)  # calculate the sign of the gradient of cost function
        eta = tf.multiply(sign_nabla_J, EPS)
        # sign_random = tf.sign(tf.random_normal([24, 24, 3], mean=0, stddev=1))
        # eta = tf.multiply(sign_random, EPS)  # multiply epsilon the sign of the gradient of cost function
        eta_reshaped = tf.reshape(eta, images_org._shape)
        images_new = tf.add(images_org, eta_reshaped)
        images_org = tf.cast(images_org, tf.float32)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                     start=True))

                step = 0
                true_count = 0
                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
                total_sample_count = num_iter * FLAGS.batch_size
                while step < num_iter and not coord.should_stop():
                    predictions, images_array_org, images_array_new, images_array, labels_array = sess.run(
                        [top_k_op, images_org, images_new, images, labels])
                    image_matrix_org = np.array([images_array_org[:, :, 0], images_array_org[:, :, 1],
                                                 images_array_org[:, :, 2]])
                    image_list_org = np.array(image_matrix_org.flatten())
                    image_list_org = image_list_org.astype('uint8')
                    image_list_org = list(image_list_org)
                    label_list_org = list(map(int, list(labels_array)))
                    data_list_org = np.array(label_list_org + image_list_org)
                    data_list_org = list(data_list_org)
                    data_bytes_org = bytes(data_list_org)

                    images_array_new = (images_array_new - 128) * 128 / (128 + EPS) + 128
                    kernel_org = [[0, 1, 0], [1, 4, 1], [0, 1, 0]]
                    kernel = np.array(kernel_org) / np.sum(kernel_org) * 1.0
                    images_array_new = cv2.filter2D(images_array_new, -1, kernel)
                    # images_array_new = cv2.blur(images_array_new, (2, 2))
                    # images_array_new = cv2.GaussianBlur(images_array_new, (3, 3), 0)

                    # img = images_array_new.astype(np.uint8)
                    # images_array_new = cv2.fastNlMeansDenoisingColored(img, None, 10, 7, 21)
                    # images_array_new = cv2.bilateralFilter(images_array_new, 9, 75, 75)

                    image_matrix_new = np.array([images_array_new[:, :, 0], images_array_new[:, :, 1],
                                                 images_array_new[:, :, 2]])
                    image_list_new = np.array(image_matrix_new.flatten())
                    image_list_new = list(image_list_new.astype('uint8'))
                    # image_list_new = (image_list_new - 128) * 128 / (128 + EPS) + 128
                    max_value = np.max(image_list_new)
                    min_value = np.min(image_list_new)
                    try:
                        assert max_value <= 255
                        assert min_value >= 0
                    except:
                        raise
                    label_list_new = list(map(int, list(labels_array)))
                    data_list_new = label_list_new + image_list_new
                    data_bytes_new = bytes(data_list_new)

                    if step == 0:
                        file_org = data_bytes_org
                        file_new = data_bytes_new
                    else:
                        file_new += data_bytes_new
                        file_org += data_bytes_org

                    if step % 200 == 0:
                        print("Step: %d" % step)
                    true_count += np.sum(predictions)
                    step += 1

                precision = true_count / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

                file_name_org = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin/test_batch_eps%d_org.bin' % EPS)
                with open(file_name_org, 'wb') as f:
                    f.write(file_org)
                file_name_noise = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin/test_batch_eps%d_noise.bin' % EPS)
                with open(file_name_noise, 'wb') as f:
                    f.write(file_new)

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

        FLAGS.batch_size = batch_size


def show_images_with_noise():
    with tf.variable_scope('network1') as scope:
        batch_size = FLAGS.batch_size
        FLAGS.image_size = 24
        FLAGS.batch_size = 1
        images, labels, images_org = cifar10.inputs(eval_data=FLAGS.eval_data)
        logits = cifar10.inference(images)

        loss = cifar10.loss(logits, labels)
        nabla_J = tf.gradients(loss, images)  # apply nabla operator to calculate the gradient
        sign_nabla_J = tf.sign(nabla_J)  # calculate the sign of the gradient of cost function
        eta = tf.multiply(sign_nabla_J, EPS)
        eta_reshaped = tf.reshape(eta, images_org._shape)
        images_new = tf.add(images_org, eta_reshaped)
        images_org = tf.cast(images_org, tf.float32)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                     start=True))

                step = 0
                true_count = 0
                num_iter = 10
                # num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
                total_sample_count = num_iter * FLAGS.batch_size
                while step < num_iter and not coord.should_stop():
                    predictions, images_array_org, images_array_new, images_array, labels_array = sess.run(
                        [top_k_op, images_org, images_new, images, labels])

                    images_dif = images_array_new - images_array_org

                    images_array_new = (images_array_new - 128) * 128 / (128 + EPS) + 128
                    kernel_org = [[0, 1, 0], [1, 4, 1], [0, 1, 0]]
                    kernel = np.array(kernel_org) / np.sum(kernel_org) * 1.0
                    images_array_new = cv2.filter2D(images_array_new, -1, kernel)
                    images_array_new = np.array(images_array_new).astype('uint8')
                    # images_array_new = cv2.blur(images_array_new, (2, 2))
                    # images_array_new = cv2.GaussianBlur(images_array_new, (3, 3), 0)

                    # img = images_array_new.astype(np.uint8)
                    # images_array_new = cv2.fastNlMeansDenoisingColored(img, None, 10, 7, 21)
                    # images_array_new = cv2.bilateralFilter(images_array_new, 9, 75, 75)

                    max_value = np.max(images_dif)
                    min_value = np.min(images_dif)
                    try:
                        assert max_value == 15
                        assert min_value == -15
                    except:
                        raise

                    plt.figure(1)
                    plt.subplot(241)
                    plt.imshow(images_array_org[:, :, 0])
                    plt.subplot(242)
                    plt.imshow(images_array_org[:, :, 1])
                    plt.subplot(243)
                    plt.imshow(images_array_org[:, :, 2])
                    plt.subplot(244)
                    plt.imshow(images_array_org)
                    plt.subplot(245)
                    plt.imshow(images_array_new[:, :, 0])
                    plt.subplot(246)
                    plt.imshow(images_array_new[:, :, 1])
                    plt.subplot(247)
                    plt.imshow(images_array_new[:, :, 1])
                    plt.subplot(248)
                    plt.imshow(images_array_new)
                    plt.show()

                    true_count += np.sum(predictions)
                    step += 1

                precision = true_count / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

        FLAGS.batch_size = batch_size


if __name__ == '__main__':
    pass
    # generate_images_size24()
    # evaluate()
