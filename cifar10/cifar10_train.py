# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf
import numpy as np

import cifar10
import data_helpers

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
MAX_STEPS = 70
IMAGE_SIZE = 24

def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.variable_scope('network1') as scope:
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        image_0 = tf.placeholder(tf.float32, shape=[3072], name='image_0')
        image_0 = tf.reshape(image_0, [32, 32, 3])
        label_0 = tf.placeholder(tf.int32, shape=[1], name='label_0')
        # images, labels = cifar10.distorted_inputs(image_0, label_0)
        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # Image processing for evaluation.
        # Crop the central [height, width] of the image.
        resized_image = tf.image.resize_image_with_crop_or_pad(image_0,
                                                               height, width)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(resized_image)
        # float_image = resized_image

        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = 20000

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(float_image)

        # Calculate loss.
        loss = cifar10.loss(logits, label_0)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = cifar10.train(loss, global_step)

        images_batch = np.ones([3072])
        images_batch = images_batch * 1.0
        labels_batch = [1]

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                     start=True))
                # sess.run(image_0, feed_dict={image_0: images_batch})
                sess.run(train_op, feed_dict={image_0: images_batch, label_0: labels_batch})
                print("test")
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
            # sess.run(train_op, feed_dict={image_0: images_batch, label_0: labels_batch})

        # class _LoggerHook(tf.train.SessionRunHook):
        #     """Logs loss and runtime."""
        #
        #     def begin(self):
        #         self._step = -1
        #         self._start_time = time.time()
        #
        #     def before_run(self, run_context):
        #         self._step += 1
        #         return tf.train.SessionRunArgs(loss)  # Asks for loss value.
        #
        #     def after_run(self, run_context, run_values):
        #         if self._step % FLAGS.log_frequency == 0:
        #             current_time = time.time()
        #             duration = current_time - self._start_time
        #             self._start_time = current_time
        #
        #             loss_value = run_values.results
        #             examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
        #             sec_per_batch = float(duration / FLAGS.log_frequency)
        #
        #             format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
        #                           'sec/batch)')
        #             print(format_str % (datetime.now(), self._step, loss_value,
        #                                 examples_per_sec, sec_per_batch))
        #
        # with tf.train.MonitoredTrainingSession(
        #         checkpoint_dir=FLAGS.train_dir,
        #         hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
        #                tf.train.NanTensorHook(loss),
        #                _LoggerHook()],
        #         config=tf.ConfigProto(
        #             log_device_placement=FLAGS.log_device_placement)) as mon_sess:
        #     data_sets = data_helpers.load_data()
        #     zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
        #     batches = data_helpers.gen_batch(list(zipped_data), 1, MAX_STEPS)
        #     images_batch = np.ones([32, 32, 3])
        #     labels_batch = [1]
        #     while not mon_sess.should_stop():
        #         # mon_sess.run(tf.global_variables_initializer())
        #         # batch = next(batches)
        #         # images_batch, labels_batch = zip(*batch)
        #         # images_batch = np.resize(images_batch, (32, 32, 3))
        #         mon_sess.run(train_op, feed_dict={label_0: labels_batch})


def main(argv=None):  # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
