#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Trainer for the Classifier model"""

import time
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from .utils import builders, flow

from .interfaces import Trainer
from ..inputs.interfaces import InputType
from ..evaluators.metrics import accuracy_op
from ..models.utils import tf_log, variables_to_train, count_trainable_parameters
from ..models.collections import MODEL_SUMMARIES
from ..models.visualization import log_io


class ClassifierTrainer(Trainer):
    """Trainer for the Classifier model"""

    def __init__(self):
        """Initialize the evaluator"""
        self._model = None

    @property
    def model(self):
        """Returns the model to evaluate"""
        return self._model

    @model.setter
    def model(self, model):
        """Set the model to evaluate.
        Args:
            model: implementation of the Model interface
        """
        self._model = model

    def train(self, dataset, args, steps, paths):
        """Train the model, using the dataset, utilizing the passed args
        Args:
            dataset: implementation of the Input interface
            args: dictionary of hyperparameters a train parameters

        Returns:
            info: dict containing the information of the trained model
        Side effect:
            saves the latest checkpoints and the best model in its own folder
        """

        with tf.Graph().as_default():
            tf.set_random_seed(69)
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Get images and labels
            with tf.device('/cpu:0'):
                images, labels = dataset.inputs(
                    input_type=InputType.train,
                    batch_size=args["batch_size"],
                    augmentation_fn=args["regularizations"]["augmentation"])
            log_io(images)

            # Build a Graph that computes the logits predictions from the
            # inference model.
            is_training_, logits = self._model.get(
                images,
                dataset.num_classes,
                train_phase=True,
                l2_penalty=args["regularizations"]["l2"])

            num_of_parameters = count_trainable_parameters(print_model=True)
            print("Model {}: trainable parameters: {}. Size: {} KB".format(
                self._model.name, num_of_parameters, num_of_parameters * 4 /
                1000))

            # Calculate loss.
            loss = self._model.loss(logits, labels)
            tf_log(tf.summary.scalar('loss', loss))

            # Create optimizer and log learning rate
            optimizer = builders.build_optimizer(args, steps, global_step)
            train_op = optimizer.minimize(
                loss,
                global_step=global_step,
                var_list=variables_to_train(args["trainable_scopes"]))

            train_accuracy = accuracy_op(logits, labels)
            # General validation summary
            accuracy_value_ = tf.placeholder(tf.float32, shape=())
            accuracy_summary = tf.summary.scalar('accuracy', accuracy_value_)

            # read collection after that every op added its own
            # summaries in the train_summaries collection
            train_summaries = tf.summary.merge(
                tf.get_collection_ref(MODEL_SUMMARIES))

            # Build an initialization operation to run below.
            init = [
                tf.variables_initializer(tf.global_variables() +
                                         tf.local_variables()),
                tf.tables_initializer()
            ]

            # Start running operations on the Graph.
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True)) as sess:
                sess.run(init)

                # Start the queue runners with a coordinator
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                # Create the savers.
                train_saver, best_saver = builders.build_train_savers(
                    [global_step])
                flow.restore_or_restart(args, paths, sess, global_step)
                train_log, validation_log = builders.build_loggers(
                    sess.graph, paths)

                # If a best model already exists (thus we're continuing a train
                # process) then restore the best validation accuracy reached
                # and place it into best_va
                best_va = self._model.evaluator.eval(
                    paths["best"],
                    dataset,
                    input_type=InputType.validation,
                    batch_size=args["batch_size"])

                # Extract previous global step value
                old_gs = sess.run(global_step)

                # Restart from where we were
                for step in range(old_gs, steps["max"] + 1):
                    start_time = time.time()
                    _, loss_value = sess.run(
                        [train_op, loss], feed_dict={is_training_: True})

                    duration = time.time() - start_time

                    if np.isnan(loss_value):
                        print('Model diverged with loss = NaN')
                        break

                    # update logs every 10 iterations
                    if step % steps["log"] == 0:
                        examples_per_sec = args["batch_size"] / duration
                        sec_per_batch = float(duration)

                        format_str = ('{}: step {}, loss = {:.4f} '
                                      '({:.1f} examples/sec; {:.3f} sec/batch)')
                        print(
                            format_str.format(datetime.now(), step, loss_value,
                                              examples_per_sec, sec_per_batch))
                        # log train values
                        summary_lines = sess.run(
                            train_summaries, feed_dict={is_training_: True})
                        train_log.add_summary(summary_lines, global_step=step)

                    # Save the model checkpoint at the end of every epoch
                    # evaluate train and validation performance
                    if (step > 0 and
                            step % steps["epoch"] == 0) or step == steps["max"]:
                        checkpoint_path = os.path.join(paths["log"],
                                                       'model.ckpt')
                        train_saver.save(
                            sess, checkpoint_path, global_step=step)

                        # validation accuracy
                        va_value = self._model.evaluator.eval(
                            paths["log"],
                            dataset,
                            input_type=InputType.validation,
                            batch_size=args["batch_size"])

                        summary_line = sess.run(
                            accuracy_summary,
                            feed_dict={accuracy_value_: va_value})
                        validation_log.add_summary(
                            summary_line, global_step=step)

                        # train accuracy
                        ta_value = sess.run(
                            train_accuracy, feed_dict={is_training_: False})
                        summary_line = sess.run(
                            accuracy_summary,
                            feed_dict={accuracy_value_: ta_value})
                        train_log.add_summary(summary_line, global_step=step)

                        print(
                            '{} ({}): train accuracy = {:.3f} validation accuracy = {:.3f}'.
                            format(datetime.now(
                            ), int(step / steps["epoch"]), ta_value, va_value))

                        # save best model
                        if va_value > best_va:
                            best_va = va_value
                            best_saver.save(
                                sess,
                                os.path.join(paths["best"], 'model.ckpt'),
                                global_step=step)
                # end of for
                validation_log.close()
                train_log.close()

                # When done, ask the threads to stop.
                coord.request_stop()
                # Wait for threads to finish.
                coord.join(threads)

            stats = self._model.evaluator.stats(
                paths["best"], dataset, batch_size=args["batch_size"])
            self._model.info = {
                "args": args,
                "paths": paths,
                "steps": steps,
                "stats": stats
            }
            return self._model.info
