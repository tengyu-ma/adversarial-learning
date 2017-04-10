from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS
EVAL_DIR = '/tmp/cifar10_eval'
EVAL_DATA = 'test'  # 'train_eval'
CHECKPOINT_DIR = '/tmp/cifar10_train'
NUM_EXAMPLES = 10000

class Cifar:
    def __init__(self):
        self.sess = tf.Session()

def evaluate(sess):
    """Eval CIFAR-10 for a number of steps."""
    # with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
    eval_data = EVAL_DATA == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data)
    images_with_noise = adversarial(images)

    # Build a Graph that computes the logits predictions from the
    # inference model.
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
    summary_writer = tf.summary.FileWriter(EVAL_DIR)

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

        num_iter = int(math.ceil(NUM_EXAMPLES / FLAGS.batch_size))
        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * FLAGS.batch_size
        step = 0
        while step < num_iter and not coord.should_stop():
            predictions = sess.run([top_k_op])
            true_count += np.sum(predictions)
            step += 1

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


def evaluate_with_noise(sess):
    print("test")


def adversarial(images):
    return images


def restore_network():
    pass


if __name__ == '__main__':
    cifar = Cifar()
    restore_network()
    evaluate(cifar.sess)
    evaluate_with_noise(cifar.sess)