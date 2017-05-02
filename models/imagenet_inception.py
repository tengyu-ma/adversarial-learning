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

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.misc import toimage

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
import cv2
from six.moves import urllib
import tensorflow as tf
import random

from utils import util
import adversarize
import denoise

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

n1_t = 0
n1_c = 0
n2_t = 0
n2_c = 0
n3_t = 0
n3_c = 0
n4_t = 0
n4_c = 0
n5_t = 0
n5_c = 0


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup, self.uid_to_id = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.
    
        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        node_uid_to_id = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]
                node_uid_to_id[target_class_string[1:-2]] = target_class

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name, node_uid_to_id

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(sess, image, image_id):
    """Runs inference on an image.
    
    Parameters
    ----------
    sess : Session
        tensorflow session
    image : str
        Image file name.
    image_id : int
        Correct label for the image
    
    Output tensors
    ______________
    'softmax:0': tensor
        A tensor containing the normalized prediction across 1000 labels.
    'pool_3:0': tensor
        A tensor containing the next-to-last layer containing 2048 float description of the image.
        
    Input tensors
    _____________
    'DecodeJpeg:0' : tensor
        the very first input tensor containing the original UINT8 image matrix
    'DecodeJpeg/contents:0': tensor
        A tensor containing a string providing JPEG encoding of the image.  
    'Cast:0' : tensor
        convert UINT8 to Float32 type for each element in the matrix
    'ExpandDims:0' : tensor
        add batch dimensions
    'ResizeBilinear:0' : tensor
        resize the image into standard 300*300 size by bilinear interpolation
    'Sub:0' : tensor
        the first step to normalize
    'Mul:0' : tensor
        the last step to normalize
      
    """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    # Creates graph from saved GraphDef.
    # create_graph()
    # with tf.Session() as sess:
    # get the output after softmax.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

    # get the output before the softmax
    softmax_logits_tensor = sess.graph.get_tensor_by_name('softmax/logits:0')

    # get the real input to the DNNs which are resized to 300*300 and normalized
    rn_image_tensor = sess.graph.get_tensor_by_name('Mul:0')

    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

    predictions_logits = sess.run(softmax_logits_tensor, {'DecodeJpeg/contents:0': image_data})

    rn_image = sess.run(rn_image_tensor, {'DecodeJpeg/contents:0': image_data})

    save_image(rn_image, 'panda1')

    predictions = np.squeeze(predictions)
    # predictions_logits = np.squeeze(predictions_logits)

    label = one_hot_label(image_id)

    J = cross_entropy_loss(tf.squeeze(softmax_logits_tensor), tf.convert_to_tensor(label),
                           label_smoothing=0.1, weight=0.4)

    epsilon = 2.0
    nabla_J = tf.gradients(J, rn_image_tensor)  # apply nabla operator to calculate the gradient
    sign_nabla_J = tf.sign(nabla_J)  # calculate the sign of the gradient of cost function
    eta = tf.multiply(sign_nabla_J, epsilon)  # multiply epsilon the sign of the gradient of cost function

    noise = sess.run(eta, {'DecodeJpeg/contents:0': image_data})
    x_adv = sess.run(tf.add(rn_image_tensor, tf.squeeze(noise)))  # add noise to test data

    predictions2 = sess.run(softmax_tensor, {'Cast:0': x_adv})
    predictions2 = np.squeeze(predictions2)

    x_adv_denoise = denoise.bilateral_filter(x_adv, epsilon)
    predictions3 = sess.run(softmax_tensor, {'Cast:0': x_adv_denoise})
    predictions3 = np.squeeze(predictions3)

    save_image(noise, 'grace_hopper_noise')
    save_image(x_adv, 'grace_hopper_adv')
    save_image(x_adv_denoise, 'grace_hopper_adv_denoise')

    global n1_t
    global n1_c
    global n2_t
    global n2_c
    global n3_t
    global n3_c
    global n4_t
    global n4_c
    global n5_t
    global n5_c

    if image_eval(predictions, image_id):
        n1_c += 1
    n1_t += 1

    if image_eval(predictions2, image_id):
        n2_c += 1
    n2_t += 1

    if image_eval(predictions3, image_id):
        n3_c += 1
    n3_t += 1

    # ========== wrong label test ============ #
    label = one_hot_label(random.sample(range(1000), 1)[0])
    J = cross_entropy_loss(tf.squeeze(softmax_logits_tensor), tf.convert_to_tensor(label),
                           label_smoothing=0.1, weight=0.4)

    # J_r = sess.run(J, {'DecodeJpeg/contents:0': image_data})

    # f = open('J_name.txt', 'w+')
    # for key, value in J.graph._names_in_use.items():
    #     f.write(key + '\t' + str(value) + '\n')
    # f.close()

    epsilon = 2
    nabla_J = tf.gradients(J, rn_image_tensor)  # apply nabla operator to calculate the gradient
    sign_nabla_J = tf.sign(nabla_J)  # calculate the sign of the gradient of cost function
    eta = tf.multiply(sign_nabla_J, epsilon)  # multiply epsilon the sign of the gradient of cost function

    noise = sess.run(eta, {'DecodeJpeg/contents:0': image_data})
    x_adv = sess.run(tf.add(rn_image_tensor, tf.squeeze(noise)))  # add noise to test data

    predictions4 = sess.run(softmax_tensor, {'Cast:0': x_adv})
    predictions4 = np.squeeze(predictions4)

    x_adv_denoise = denoise.bilateral_filter(x_adv, epsilon)
    predictions5 = sess.run(softmax_tensor, {'Cast:0': x_adv_denoise})
    predictions5 = np.squeeze(predictions5)

    if image_eval(predictions4, image_id):
        n4_c += 1
    n4_t += 1

    if image_eval(predictions5, image_id):
        n5_c += 1
    n5_t += 1

    print(n1_c/n1_t, '\t', n2_c/n2_t, '\t', n3_c/n3_t, '\t', n4_c/n4_t, '\t', n5_c/n5_t, '\n')


# def get_nom_image


def image_eval(prediciton, image_id):
    # Creates node ID --> English string lookup.
    # f = open('eval_o_a_d.txt', 'a+')
    node_lookup = NodeLookup()
    top_k = prediciton.argsort()[-FLAGS.num_top_predictions:][::-1]

    for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        score = prediciton[node_id]
        # if node_id == image_id:
        print('%s (score = %.5f)' % (human_string, score))
            # return True
        # print('wrong! %s (score = %.5f)' % (human_string, score))
    # return False

    #     f.write('%d\t%s (score = %.5f)\n' % (i, human_string, score))
    # f.write('\n')
    # f.close()
    # print('\n')


def one_hot_label(entry):
    one_hot = np.zeros((1008,), dtype=np.float32)
    one_hot[entry] = 1
    return one_hot


def cross_entropy_loss(logits, one_hot_labels, label_smoothing=0.0,
                       weight=1.0, scope=None):
    """Define a Cross Entropy loss using softmax_cross_entropy_with_logits.

    It can scale the loss by weight factor, and smooth the labels.

    Args:
        logits: [num_classes] logits outputs of the network .
        one_hot_labels: [num_classes] target one_hot_encoded labels.
        label_smoothing: if greater than 0 then smooth the labels.
        weight: scale the loss by this factor.
        scope: Optional scope for name_scope.

    Returns:
        A tensor with the softmax_cross_entropy loss.
    """

    logits.get_shape().assert_is_compatible_with(one_hot_labels.get_shape())
    with tf.name_scope(scope, 'CrossEntropyLoss', [logits, one_hot_labels]):
        num_classes = one_hot_labels.get_shape()[-1].value
        one_hot_labels = tf.cast(one_hot_labels, logits.dtype)
        if label_smoothing > 0:
            smooth_positives = 1.0 - label_smoothing
            smooth_negatives = label_smoothing / num_classes
            one_hot_labels = one_hot_labels * smooth_positives + smooth_negatives
        cross_entropy = tf.contrib.nn.deprecated_flipped_softmax_cross_entropy_with_logits(
            logits, one_hot_labels, name='xentropy')

        weight = tf.convert_to_tensor(weight,
                                      dtype=logits.dtype.base_dtype,
                                      name='loss_weight')
        loss = tf.multiply(weight, tf.reduce_mean(cross_entropy), name='value')
        return loss


def save_image(image_matrix, label):
    image_matrix = np.squeeze(image_matrix)
    # toimage(image_matrix, cmin=0.0, cmax=1.0).save('%s.jpg' % label)
    save_path = os.path.join(util.ROOT_DIR, "data", "imagenet", "%s.jpg" % label)
    toimage(image_matrix).save(save_path)


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


class InceptionModel:
    def __init__(self, num_top_predictions=5):
        self.model_dir = os.path.join(util.ROOT_DIR, "data/imagenet/nom")
        self.num_top_predictions = num_top_predictions

FLAGS = InceptionModel()


def main(_):
    # maybe_download_and_extract()
    single = True  # run inference on a single image or a batch of images
    sess = tf.Session()
    if single:
        label = 169  # 169 for panda; 866 for hobber
        image = (os.path.join(FLAGS.model_dir, FLAGS.image_file))
        run_inference_on_image(sess, image, label)
    else:
        image_path = "E:/tmp/mydata/raw-data/validation"
        for root, dirs, files in os.walk(image_path):
            for file in files:
                image = os.path.join(root, file)
                uid = root.split('\\')[-1]
                node_lookup = NodeLookup()
                label = node_lookup.uid_to_id[uid]
                run_inference_on_image(sess, image, label)


def make_prediction():
    """
    classify_image_graph_def.pb:
      Binary representation of the GraphDef protocol buffer.
    imagenet_synset_to_human_label_map.txt:
      Map from synset ID to a human readable string.
    imagenet_2012_challenge_label_map_proto.pbtxt:
      Text representation of a protocol buffer mapping a label to synset ID.
    """
    # argv = [sys.argv[0]]
    tf.app.run(main=main)
