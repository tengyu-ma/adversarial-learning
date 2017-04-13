from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os

import numpy as np
import tensorflow as tf
import matplotlib as mp
import matplotlib.pyplot as plt

import cifar10
import data_helpers

FLAGS = tf.app.flags.FLAGS
EVAL_DIR = '/tmp/cifar10_eval'
EVAL_DATA = 'test'  # 'train_eval'
CHECKPOINT_DIR = '/tmp/cifar10_train'
NUM_EXAMPLES = 10000
EPS = 5
MAX_STEPS = 70


class Cifar:
    def __init__(self):
        self.sess = tf.Session()
        self.image_noise = None

    def evaluate(self):
        """Eval CIFAR-10 for a number of steps."""
        with tf.variable_scope('network1') as scope:
            # Get images and labels for CIFAR-10.
            eval_data = EVAL_DATA == 'test'
            data_sets = data_helpers.load_data()

            images, labels, org_images = cifar10.inputs(eval_data=eval_data)
            org_images = tf.cast(org_images, tf.float32)

            logits = cifar10.inference(images)

            loss = cifar10.loss(logits, labels)
            nabla_J = tf.gradients(loss, images)  # apply nabla operator to calculate the gradient
            sign_nabla_J = tf.sign(nabla_J)  # calculate the sign of the gradient of cost function
            eta = tf.multiply(sign_nabla_J, EPS)  # multiply epsilon the sign of the gradient of cost function
            eta_reshaped = tf.reshape(eta, images._shape)
            images_new = tf.add(org_images, eta_reshaped)

            # scope.reuse_variables()
            # logits = cifar10.inference(images_new)

            # Calculate predictions.
            top_k_op = tf.nn.in_top_k(logits, labels, 1)

            # Restore the moving average version of the learned variables for eval.
            variable_averages = tf.train.ExponentialMovingAverage(
                cifar10.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()

            summary_writer = tf.summary.FileWriter(EVAL_DIR)

            with tf.Session() as sess:

                ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # Assuming model_checkpoint_path looks something like:
                    #   /my-favorite-path/cifar10_train/model.ckpt-0,
                    # extract global_step from it.
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                else:
                    print('No checkpoint file found')
                    return

                # Start the queue runners.
                coord = tf.train.Coordinator()
                try:
                    threads = []
                    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                         start=True))

                    # num_iter = int(math.ceil(NUM_EXAMPLES / FLAGS.batch_size))
                    num_iter = 1000
                    true_count = 0  # Counts the number of correct predictions.
                    total_sample_count = num_iter * FLAGS.batch_size
                    step = 0
                    file = None
                    while step < num_iter and not coord.should_stop():
                        # org_images_4d, images_4d, eta_4d, images_new_4d = sess.run(
                        #     [org_images, images, eta_reshaped, images_new])
                        predictions, org_images_array, images_array, labels_array = sess.run([top_k_op, images_new, images, labels])
                        # predictions= sess.run([top_k_op])
                        print("Step: %d" % step)
                        true_count += np.sum(predictions)
                        if FLAGS.batch_size == 1:
                        #     plt.imshow(org_images_array)
                        #     plt.show()
                            image_matrix = np.array([org_images_array[0,:,:,0], org_images_array[0,:,:,1], org_images_array[0,:,:,2]])
                            image_list = list(map(int, list(image_matrix.flatten())))
                            label_list = list(map(int, list(labels_array)))
                            data_list = np.array(label_list + image_list)
                            data_list[data_list > 255] = 255
                            data_list[data_list < 0] = 0
                            data_list = list(data_list)
                            data_bytes = bytes(data_list)
                            assert len(data_bytes) == 1729
                            if step == 0:
                                file = data_bytes
                            else:
                                file += data_bytes

                        step += 1

                    data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
                    file_to_write = os.path.join(data_dir, 'test_batch_new.bin')
                    with open(file_to_write, 'wb') as f:
                        f.write(file)

                    # scale = 128.0 / max(abs(tmp.max()),abs(tmp.min()))
                    # tmp = tmp * scale + 128.0
                    # imgplot = plt.imshow(tmp)
                    # plt.show()

                    # Compute precision @ 1.
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

    def evaluate_with_noise(self):
        eval_data = EVAL_DATA == 'test'
        images, labels = cifar10.inputs(eval_data=eval_data)
        logits = cifar10.inference(self.image_noise)

    def adversarial(self, images):
        return images

    def restore_network(self):
        pass


if __name__ == '__main__':
    cifar = Cifar()
    cifar.restore_network()
    # with tf.device('/gpu:0'):
    cifar.evaluate()
    # cifar.evaluate_with_noise()
