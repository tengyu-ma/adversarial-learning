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

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

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

                precision = true_count / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1', simple_value=precision)
                summary_writer.add_summary(summary, global_step)

                file_to_write_org = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin\test_batch_size24.bin')
                with open(file_to_write_org, 'wb') as f:
                    f.write(file_org)

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

    FLAGS.batch_size = batch_size


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.variable_scope('network1') as scope:
        # Get images and labels for CIFAR-10.
        eval_data = FLAGS.eval_data == 'test'

        images, labels, images_org = cifar10.inputs(eval_data=eval_data)

        logits = cifar10.inference(images)

        if MODE == '24_to_noise' or MODE == 'show' or MODE == 'vote_show':  # we need to save the image with noise
            assert ORG_IMAGE_SIZE == 24
            loss = cifar10.loss(logits, labels)
            nabla_J = tf.gradients(loss, images)  # apply nabla operator to calculate the gradient
            sign_nabla_J = tf.sign(nabla_J)  # calculate the sign of the gradient of cost function
            eta = tf.multiply(sign_nabla_J, EPS)
            # sign_random = tf.sign(tf.random_normal([24, 24, 3], mean=0, stddev=1))
            # eta = tf.multiply(sign_random, EPS)  # multiply epsilon the sign of the gradient of cost function
            eta_reshaped = tf.reshape(eta, images_org._shape)
            images_new = tf.add(images_org, eta_reshaped)
            images_org = tf.cast(images_org, tf.float32)

        if MODE == 'vote_show':
            scope.reuse_variables()
            eta_0 = tf.multiply(sign_nabla_J, 0.1)
            eta_r0 = tf.reshape(eta_0, images._shape)
            images_n0 = tf.add(images, eta_r0)
            logits_0 = cifar10.inference(images_n0)

            sign_random_1 = tf.sign(tf.random_normal([24, 24, 3], mean=0, stddev=1))
            eta_1 = tf.multiply(sign_random_1, 0.20)
            eta_r1 = tf.reshape(eta_1, images._shape)
            images_n1 = tf.add(images_n0, eta_r1)
            logits_1 = cifar10.inference(images_n1)

            sign_random_2 = tf.sign(tf.random_normal([24, 24, 3], mean=0, stddev=1))
            eta_2 = tf.multiply(sign_random_2, 0.25)
            eta_r2 = tf.reshape(eta_2, images._shape)
            images_n2 = tf.add(images_n0, eta_r2)
            logits_2 = cifar10.inference(images_n2)

            sign_random_3 = tf.sign(tf.random_normal([24, 24, 3], mean=0, stddev=1))
            eta_3 = tf.multiply(sign_random_3, 0.30)
            eta_r3 = tf.reshape(eta_3, images._shape)
            images_n3 = tf.add(images_n0, eta_r3)
            logits_3 = cifar10.inference(images_n3)

            sign_random_4 = tf.sign(tf.random_normal([24, 24, 3], mean=0, stddev=1))
            eta_4 = tf.multiply(sign_random_4, 0.35)
            eta_r4 = tf.reshape(eta_4, images._shape)
            images_n4 = tf.add(images_n0, eta_r4)
            logits_4 = cifar10.inference(images_n4)

            sign_random_5 = tf.sign(tf.random_normal([24, 24, 3], mean=0, stddev=1))
            eta_5 = tf.multiply(sign_random_5, 0.40)
            eta_r5 = tf.reshape(eta_5, images._shape)
            images_n5 = tf.add(images_n0, eta_r5)
            logits_5 = cifar10.inference(images_n5)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

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

                # num_iter = 1000
                num_iter = int(math.ceil(NUM_EXAMPLES / FLAGS.batch_size))
                if MODE == 'show' or MODE == 'vote_show':
                    num_iter = 10
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0
                file_org = None
                file_new = None
                while step < num_iter and not coord.should_stop():
                    if MODE == '24_to_noise' or MODE == 'show':
                        predictions, images_array_org, images_array_new, images_array, labels_array = sess.run(
                            [top_k_op, images_org, images_new, images, labels])
                        # predictions = sess.run([top_k_op])
                    elif MODE == 'vote_show':
                        predictions, images_array_org, images_array_new, images_array, labels_array, \
                        l_num, l_num0, l_num1, l_num2, l_num3, l_num4, l_num5 = sess.run(
                            [top_k_op, images_org, images_new, images, labels,
                             logits, logits_0, logits_1, logits_2, logits_3, logits_4, logits_5])
                    elif MODE == '32_to_24':
                        predictions, images_array_org, images_array, labels_array = sess.run(
                            [top_k_op, images_org, images, labels])
                    else:
                        predictions = sess.run([top_k_op])

                    if MODE == '24_to_noise' or MODE == '32_to_24':
                        image_matrix_org = np.array([images_array_org[:, :, 0], images_array_org[:, :, 1],
                                                     images_array_org[:, :, 2]])
                        image_list_org = np.array(image_matrix_org.flatten())
                        image_list_org = image_list_org.astype('uint8')
                        # image_list_org = (image_list_org - 127.5) * 255 / (255 + 2 * EPS) + 127.5
                        image_list_org = list(image_list_org)
                        label_list_org = list(map(int, list(labels_array)))
                        data_list_org = np.array(label_list_org + image_list_org)
                        data_list_org = list(data_list_org)
                        data_bytes_org = bytes(data_list_org)

                        if step == 0:
                            file_org = data_bytes_org
                        else:
                            file_org += data_bytes_org

                    if MODE == '24_to_noise':
                        images_array_new = (images_array_new - 128) * 128 / (128 + EPS) + 128
                        # images_array_new = 255 - images_array_new
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
                            file_new = data_bytes_new
                        else:
                            file_new += data_bytes_new

                    if MODE == 'show':
                        image_list_new = np.array(images_array_new.flatten())
                        image_list_new = (image_list_new - 128) * 128 / (128 + EPS) + 128
                        image_list_new = list(map(float, map(math.floor, list(image_list_new))))
                        images_array_new = np.reshape(image_list_new, (24, 24, 3))

                        kernel_org = [[0, 1, 0], [1, 4, 1], [0, 1, 0]]
                        kernel = np.array(kernel_org) / np.sum(kernel_org) * 1.0
                        images_array_new = cv2.filter2D(images_array_new, -1, kernel)
                        # images_array_new = cv2.bilateralFilter(images_array_new, 9, 50, 50)
                        # z = pybm3d.bm3d(images_array_new, 40.0)
                        # img = images_array_new.astype(np.uint8)
                        # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # to be gray scale
                        # img_res = [np.zeros((24, 24, 3))]
                        # res = cv2.denoise_TVL1(img, img_res[0])
                        # res = cv2.fastNlMeansDenoisingColored(img, None, 10, 7, 21)
                        # images_array_new = cv2.blur(images_array_new, (2, 2))
                        # images_array_new = cv2.GaussianBlur(images_array_new, (3, 3), 0)

                        for i in range(24):
                            for j in range(24):
                                for k in range(3):
                                    images_array_new[i][j][k] = int(images_array_new[i][j][k])

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

                    if MODE == 'vote_show':
                        print(labels_array)
                        print(np.argmax(l_num), l_num)
                        print(np.argmax(l_num0), l_num0)
                        print(np.argmax(l_num1), l_num1)
                        print(np.argmax(l_num2), l_num2)
                        print(np.argmax(l_num3), l_num3)
                        print(np.argmax(l_num4), l_num4)
                        print(np.argmax(l_num5), l_num5)
                        plt.imshow(images_array_org)
                        plt.show()
                        # plt.imshow(images)
                        # plt.show()

                    print("Step: %d" % step)
                    true_count += np.sum(predictions)
                    step += 1

                if MODE == '32_to_24':
                    file_to_write_org = os.path.join(FLAGS.data_dir, 'test_batch_org.bin')
                    with open(file_to_write_org, 'wb') as f:
                        f.write(file_org)
                # if MODE == '24_to_noise':
                #     file_to_write_org = os.path.join(DATA_DIR, 'test_batch_org_custom_%d.bin' % EPS)
                #     with open(file_to_write_org, 'wb') as f:
                #         f.write(file_org)
                #     file_to_write_new = os.path.join(DATA_DIR, 'test_batch_noise_custom_%d.bin' % EPS)
                #     with open(file_to_write_new, 'wb') as f:
                #         f.write(file_new)

                precision = true_count / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1', simple_value=precision)
                summary_writer.add_summary(summary, global_step)
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

def adversarial(self, images):
    return images

def restore_network(self):
    pass


if __name__ == '__main__':
    generate_images_size24()
    # evaluate()
