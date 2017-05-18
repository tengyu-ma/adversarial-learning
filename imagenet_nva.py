from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np
import adversarize
import denoise
import models.imagenet_inception as icp

from models.imagenet.inception.image_processing import inputs
from models.imagenet.inception import inception_eval
from models.imagenet.inception.imagenet_data import ImagenetData
from scipy.misc import toimage
from utils import util


class ImageNetNvA:
    """Imagenet normal vs adversarial class
    
    Attributes
    __________
    sess : Session
        tensorflow session
    x : placeholder
        tensorflow placeholder for the input data
    y_ : placeholder
        tensorflow placeholder for the correct label
    rn_x : tensor
        tensorflow tensor for the resized and normalized input data
    y : tensor
        tensorflow tensor of predicted value by InceptionV3
    J : tensor
        tensorflow tensor of cost function of InceptionV3
    image_data : 
        binary single input image data
    image_label : ndarray
        one hot correct label
        
    """

    def __init__(self):
        # Creates graph from saved GraphDef.
        icp.create_graph()
        self.sess = tf.Session()
        self.image_data = None
        self.image_label = None
        self.x = self.sess.graph.get_tensor_by_name('DecodeJpeg/contents:0')
        self.y_ = tf.placeholder(tf.float32, shape=[1008, ])
        self.rn_x = self.sess.graph.get_tensor_by_name('Mul:0')
        self.y = self.sess.graph.get_tensor_by_name('softmax:0')
        softmax_logits_tensor = self.sess.graph.get_tensor_by_name('softmax/logits:0')
        self.J = icp.cross_entropy_loss(tf.squeeze(softmax_logits_tensor),
                                        self.y_,
                                        label_smoothing=0.1, weight=0.4)

    def get_rn_image(self):
        """Get the resized normalized image tensor for the original image.
        Size: 300*300*3
        
        Returns
        -------
        rn_image : ndarray
            the 300*300*3 image matrix of the resized and normalized image

        """
        rn_image = self.sess.run(self.rn_x, {self.x: self.image_data})
        return rn_image

    def adversarize(self, adv_name, eps=2):
        """Add adversarial noise to the resized and normalized data
        
        Parameters
        ----------
        adv_name : str, 'fgsm'
            the method name to add adversarial noise
        eps : float
            adversarial rate

        Returns
        -------
        adv_image : ndarray
            the resized and normalized image adding adversarial noise
        adv_label : ndarray
            the predicted label by InceptionV3 for the adversarial example
        noise : ndarray
            the noise matrix

        """
        if adv_name == 'fgsm':
            adv_image, noise = adversarize.fgsm_imagenet(self.sess,
                                                         self.J,
                                                         self.x, self.y_,
                                                         self.rn_x,
                                                         self.image_data, self.image_label,
                                                         epsilon=eps)
            adv_label = self.sess.run(self.y, {self.rn_x: adv_image})
        else:
            raise Exception("Unknown adversarial method: %s \n" % adv_name)
        return adv_image, adv_label, noise

    def adversarize_batch(self, x_adv_tensor, eta, image, label, adv_name, eps=0.007):
        """Add adversarial noise to the resized and normalized data

        Parameters
        ----------
        x_adv_tensor : tensor
            tensor for adversarial image
        eta : tensor
            tensor for adversarial noise
        image : ndarray
            the regularized and normalized image data
        label : ndarray, shape=(1,)
            the correct label for the image
        adv_name : str, 'fgsm'
            the method name to add adversarial noise
        eps : float
            adversarial rate

        Returns
        -------
        adv_image : ndarray
            the resized and normalized image adding adversarial noise
        adv_label : ndarray
            the predicted label by InceptionV3 for the adversarial example
        noise : ndarray
            the noise matrix

        """
        label = icp.one_hot_label(label)
        if adv_name == 'fgsm':
            adv_image, noise = adversarize.fgsm_imagenet_batch(self.sess,
                                                               x_adv_tensor,
                                                               eta,
                                                               self.J,
                                                               self.x, self.y_,
                                                               self.rn_x,
                                                               image, label,
                                                               epsilon=eps)
            adv_label = self.sess.run(self.y, {self.rn_x: adv_image})
            # adv_label = None
        elif adv_name == 'random':
            adv_image, noise = adversarize.random_imagenet_batch(self.J,
                                                                 self.x, self.y_,
                                                                 self.rn_x,
                                                                 image, label,
                                                                 epsilon=eps)
            adv_label = self.sess.run(self.y, {self.rn_x: adv_image})
        else:
            raise Exception("Unknown adversarial method: %s \n" % adv_name)
        return adv_image, adv_label, noise

    def adversarize_batch_init(self, adv_name, eps=0.007):
        """Add adversarial noise to the resized and normalized data

        Parameters
        ----------
        adv_name : str, 'fgsm'
            the method name to add adversarial noise
        eps : float
            adversarial rate

        Returns
        -------
        adv_image : tensor
            tensor for adversarial image
        eta : tensor
            tensor for adversarial noise

        """
        # label = icp.one_hot_label(label)
        if adv_name == 'fgsm':
            adv_tensor, eta = adversarize.fgsm_imagenet_batch_init(self.J,
                                                                   self.rn_x,
                                                                   epsilon=eps)
            # adv_label = self.sess.run(self.y, {self.rn_x: adv_image})
            # adv_label = None
        elif adv_name == 'random':
            return None, None
        else:
            raise Exception("Unknown adversarial method: %s \n" % adv_name)
        return adv_tensor, eta

    def counter_fgsm_batch_init(self, denoise_name, eps=0.007):
        """Add adversarial noise to the resized and normalized data

        Parameters
        ----------
        adv_name : str, 'fgsm'
            the method name to add adversarial noise
        eps : float
            adversarial rate

        Returns
        -------
        adv_image : tensor
            tensor for adversarial image
        eta : tensor
            tensor for adversarial noise

        """
        # label = icp.one_hot_label(label)
        if denoise_name == 'counter_fgsm':
            counter_adv_tensor, eta = denoise.counter_fgsm_imagenet_batch_init(self.J,
                                                                               self.rn_x,
                                                                               epsilon=eps)
        else:
            raise Exception("Unknown denoise method: %s \n" % denoise_name)
        return counter_adv_tensor, eta

    def denoise(self, adv_image, denoise_name):
        """Denoise adversarial image
        
        Parameters
        ----------
        adv_image : ndarray
            the adversarial example of input image
        denoise_name : str
            the method used to denoise adversarial example

        Returns
        -------
        denoise_image : adarray
            the denoise image
        denoise_label : adarray
            one-hot label predicted by InceptionV3 for the denoise example
            
        """
        if denoise_name == 'bilateral':
            denoise_image = denoise.bilateral_filter(adv_image, 9, 20, 20)
        elif denoise_name == 'erosion':
            denoise_image = denoise.erosion_imagenet(adv_image)
        elif denoise_name == 'dilation':
            denoise_image = denoise.dilation_imagenet(adv_image)
        elif denoise_name == 'opening':
            denoise_image = denoise.opening_imagenet(adv_image)
        elif denoise_name == 'closing':
            denoise_image = denoise.closing_imagenet(adv_image)
        elif denoise_name == 'gradient':
            denoise_image = denoise.gradient_imagenet(adv_image)
        elif denoise_name == 'equalizeHist':
            denoise_image = denoise.equalizeHist_imagenet(adv_image)
        elif denoise_name == 'gaussian':
            denoise_image = denoise.guassian_imagenet(adv_image)
        elif denoise_name == 'contrast':
            denoise_image = denoise.contrast(adv_image)
        elif denoise_name == 'random_cover':
            denoise_image = denoise.random_cover_imagenet(adv_image, epsilon=0.007)
        elif denoise_name == 'bilateral_cover':
            denoise_image = denoise.bilateral_and_cover(adv_image)
        else:
            raise Exception("Unknown denoise method: %s \n" % denoise_name)

        denoise_label = self.sess.run(self.y, {self.rn_x: denoise_image})
        return denoise_image, denoise_label

    def counter_fgsm_denoise(self,
                             counter_x_adv_tensor, counter_eta,
                             image, label, eps=0.007):
        denoise_image, noise = denoise.counter_fgsm(self.sess,
                                                    counter_x_adv_tensor,
                                                    counter_eta,
                                                    self.J,
                                                    self.x, self.y_,
                                                    self.rn_x,
                                                    image, label,
                                                    epsilon=eps)
        denoise_label = self.sess.run(self.y, {self.rn_x: denoise_image})
        return denoise_image, denoise_label

    def human_readable_result(self, predictions):
        """Concert predictions id list into human readable string
        
        Parameters
        ----------
        predictions : ndarray
            the list of predicstions id

        """
        predictions = np.squeeze(predictions)
        node_lookup = icp.NodeLookup()
        top_k = predictions.argsort()[-icp.FLAGS.num_top_predictions:][::-1]
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

    def image_eval(self, predictions, top_n):
        top_k = predictions.argsort()[-top_n:][::-1]
        if np.argmax(self.image_label) in top_k:
            return True

    def nom_test(self):
        nom_image, nom_label = self.single_inference()

        print('* Normal test:')
        self.human_readable_result(nom_label)

        nom_label = np.squeeze(nom_label)
        node_lookup = icp.NodeLookup()
        node_id = nom_label.argsort()[-1:][::-1][0]
        human_string = node_lookup.id_to_string(node_id)
        score = nom_label[node_id]

        if self.image_eval(nom_label, 1):
            print('Correct! %s (score = %.5f)' % (human_string, score))
        else:
            print('Wrong! %s (score = %.5f)' % (human_string, score))

        self.save_image(nom_image, 'nom ' + human_string)

    def adv_test(self, adv_name='fgsm', eps=2):
        """Adversarial test for the input data
        
        Parameters
        ----------
        adv_name : str, 'fgsm'
            the method name to add adversarial noise
        eps : float
            adversarial rate

        """
        adv_image, adv_label, noise = self.adversarize(adv_name, eps)

        print('* Adversarial test:')
        self.human_readable_result(adv_label)

        adv_label = np.squeeze(adv_label)
        node_lookup = icp.NodeLookup()
        node_id = adv_label.argsort()[-1:][::-1][0]
        human_string = node_lookup.id_to_string(node_id)
        score = adv_label[node_id]

        if self.image_eval(adv_label, 1):
            print('Correct! %s (score = %.5f)' % (human_string, score))
        else:
            print('Wrong! %s (score = %.5f)' % (human_string, score))

        self.save_image(adv_image, 'adv ' + human_string)
        self.save_image(noise, 'noise ' + human_string)

    def denoised_test(self, adv_name='fgsm', denoise_name='bilateral', eps=0.007):
        """Denoise test for the input data

        Parameters
        ----------
        adv_name : str, 'fgsm'
            the method name to add adversarial noise
        denoise_name : str, 'bilateral'
            the method name to denosie adversarial example
        eps : float
            adversarial rate

        """
        adv_image, adv_label, noise = self.adversarize(adv_name, eps)
        denoise_image, denoise_label = self.denoise(adv_image, denoise_name)

        print('* Denoise test:')
        self.human_readable_result(denoise_label)

        denoise_label = np.squeeze(denoise_label)
        node_lookup = icp.NodeLookup()
        node_id = denoise_label.argsort()[-1:][::-1][0]
        human_string = node_lookup.id_to_string(node_id)
        score = denoise_label[node_id]

        if self.image_eval(adv_label, 1):
            print('Correct! %s (score = %.5f)' % (human_string, score))
        else:
            print('Wrong! %s (score = %.5f)' % (human_string, score))

        self.save_image(denoise_image, 'denoise ' + human_string)

    def set_image_file(self, single_image_name, label):
        """Assign image data and corresponding label to the class.
        
        Parameters
        ----------
        single_image_name : str
            the name of the image
        label : int
            the id of the image

        """
        image_path = self.single_image_path(single_image_name)
        if not tf.gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        self.image_data = image_data
        self.image_label = icp.one_hot_label(label)

    def single_inference(self):
        """Run single image prediction on InceptionV3

        Parameters
        ----------
        image_file : the file name of a predicting image
        image_label : the correct label of a predicting image

        """
        assert self.image_data is not None, "The input image data can not be None"
        nom_image = self.sess.run(self.rn_x, {self.x: self.image_data})
        predictions = self.sess.run(self.y, {self.x: self.image_data})
        # self.human_readable_result(predictions)
        return nom_image, predictions

    def set_batch_images(self):
        """Assign batches of image data in validation dataset and corresponding labels to the class.
        data store at E:/tmp/validation

        """
        dataset = ImagenetData(subset=icp.FLAGS.subset)
        assert dataset.data_files()
        if tf.gfile.Exists(icp.FLAGS.eval_dir):
            tf.gfile.DeleteRecursively(icp.FLAGS.eval_dir)
        tf.gfile.MakeDirs(icp.FLAGS.eval_dir)
        self.image_data = dataset

    def batch_inference(self, val_num=1000):
        images, labels, synset, text, name, table = inputs(self.image_data, batch_size=1)

        top_1_op = tf.nn.in_top_k(self.y, labels, 1)
        top_5_op = tf.nn.in_top_k(self.y, labels, 5)
        table.init.run(session=self.sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        counter = 0
        count_top_1 = 0.0
        count_top_5 = 0.0
        total_sample_count = 0.0
        total_conf = 0.0

        while not coord.should_stop():
            images_data = self.sess.run([images,
                                         labels])
            top_1 = self.sess.run(top_1_op, {self.rn_x: images_data[0],
                                             labels: images_data[1]})
            top_5 = self.sess.run(top_5_op, {self.rn_x: images_data[0],
                                             labels: images_data[1]})

            count_top_1 += np.sum(top_1)
            count_top_5 += np.sum(top_5)
            total_sample_count += 1

            predictions = self.sess.run(self.y, {self.rn_x: images_data[0]})
            predictions = np.squeeze(predictions)
            node_id = predictions.argsort()[-1:][::-1][0]
            score = predictions[node_id]
            total_conf += score

            # predictions = self.sess.run(self.y, {self.rn_x: images_data[0]})
            # node_lookup = icp.NodeLookup()
            # predictions = np.squeeze(predictions)
            # node_id = predictions.argsort()[-1:][::-1][0]
            # human_string = node_lookup.id_to_string(node_id)
            # score = predictions[node_id]
            # self.save_image(images_data[0], 'batch ' + human_string)

            counter += 1
            if counter % 100 == 0:
                precision_at_1 = count_top_1 / total_sample_count
                conf_at_1 = total_conf / total_sample_count
                recall_at_5 = count_top_5 / total_sample_count

                print("=== Step %d ===" % counter)
                print("top 1 accuracy: %.4f" % precision_at_1)
                print("top 1 confidence: %.4f" % conf_at_1)
                print("top 5 accuracy: %.4f" % recall_at_5)

            if counter == val_num:
                coord.request_stop()
        coord.join(threads)

    def batch_adv_inference(self, val_num=1000, adv_name='fgsm', eps=0.007):

        images, labels, synset, text, name, table = inputs(self.image_data, batch_size=1)

        top_1_op = tf.nn.in_top_k(self.y, labels, 1)
        top_5_op = tf.nn.in_top_k(self.y, labels, 5)
        table.init.run(session=self.sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        x_adv_tensor, eta = self.adversarize_batch_init(adv_name, eps)

        counter = 0
        count_top_1 = 0.0
        count_top_5 = 0.0
        total_sample_count = 0.0
        total_conf = 0.0

        f = open('fgsm_subtract.txt', 'w+')
        while not coord.should_stop():
            images_data = self.sess.run([images,
                                         labels])

            adv_image, adv_label, noise = self.adversarize_batch(x_adv_tensor,
                                                                 eta,
                                                                 images_data[0],
                                                                 images_data[1],
                                                                 adv_name, eps)

            top_1 = self.sess.run(top_1_op, {self.rn_x: adv_image,
                                             labels: images_data[1]})
            top_5 = self.sess.run(top_5_op, {self.rn_x: adv_image,
                                             labels: images_data[1]})

            count_top_1 += np.sum(top_1)
            count_top_5 += np.sum(top_5)
            total_sample_count += 1

            predictions = self.sess.run(self.y, {self.rn_x: adv_image})
            predictions = np.squeeze(predictions)
            node_id = predictions.argsort()[-1:][::-1][0]
            score = predictions[node_id]
            total_conf += score

            # node_lookup = icp.NodeLookup()
            # node_id = images_data[1][0]
            # human_string = node_lookup.id_to_string(node_id)
            # self.save_image(images_data[0], 'batch_nom ' + human_string)
            #
            # predictions = adv_label
            # predictions = np.squeeze(predictions)
            # node_id = predictions.argsort()[-1:][::-1][0]
            # human_string = node_lookup.id_to_string(node_id)
            # # score = predictions[node_id]
            # self.save_image(adv_image, 'batch_adv ' + human_string)

            counter += 1
            if counter % 100 == 0:
                precision_at_1 = count_top_1 / total_sample_count
                conf_at_1 = total_conf / total_sample_count
                recall_at_5 = count_top_5 / total_sample_count

                print("=== Step %d ===" % counter)
                print("top 1 accuracy: %.4f" % precision_at_1)
                print("top 1 confidence: %.4f" % conf_at_1)
                print("top 5 accuracy: %.4f" % recall_at_5)
                f.write("%d\t%.4f\t%.4f\t%.4f\n" % (counter, precision_at_1, conf_at_1, recall_at_5))

            if counter == val_num:
                coord.request_stop()
        coord.join(threads)

    def batch_denoise_inference(self, val_num=1000, adv_name='fgsm', eps=0.007, denoise_name='bilateral'):

        images, labels, synset, text, name, table = inputs(self.image_data, batch_size=1)

        top_1_op = tf.nn.in_top_k(self.y, labels, 1)
        top_5_op = tf.nn.in_top_k(self.y, labels, 5)
        table.init.run(session=self.sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        x_adv_tensor, eta = self.adversarize_batch_init(adv_name, eps)
        if denoise_name == 'counter_fgsm':
            x_counter_adv_tensor, counter_eta = self.counter_fgsm_batch_init(denoise_name, eps)

        counter = 0
        count_top_1 = 0.0
        count_top_5 = 0.0
        total_sample_count = 0.0
        total_conf = 0.0

        f = open('%s.txt' % denoise_name, 'w+')
        while not coord.should_stop():
            images_data = self.sess.run([images,
                                         labels])

            adv_image, adv_label, noise = self.adversarize_batch(x_adv_tensor,
                                                                 eta,
                                                                 images_data[0],
                                                                 images_data[1],
                                                                 adv_name, eps)
            if denoise_name == 'counter_fgsm':
                denoise_image, denoise_label = self.counter_fgsm_denoise(
                    x_counter_adv_tensor, counter_eta,
                    adv_image, np.squeeze(adv_label)
                )
            else:
                denoise_image, denoise_label = self.denoise(adv_image, denoise_name)

            top_1 = self.sess.run(top_1_op, {self.rn_x: denoise_image,
                                             labels: images_data[1]})
            top_5 = self.sess.run(top_5_op, {self.rn_x: denoise_image,
                                             labels: images_data[1]})

            count_top_1 += np.sum(top_1)
            count_top_5 += np.sum(top_5)
            total_sample_count += 1

            predictions = self.sess.run(self.y, {self.rn_x: denoise_image})
            predictions = np.squeeze(predictions)
            node_id = predictions.argsort()[-1:][::-1][0]
            score = predictions[node_id]
            total_conf += score

            save_image = False
            if save_image:
                node_lookup = icp.NodeLookup()
                node_id = images_data[1][0]
                human_string = node_lookup.id_to_string(node_id)
                self.save_image(images_data[0], 'batch_nom ' + human_string)

                predictions = adv_label
                predictions = np.squeeze(predictions)
                node_id = predictions.argsort()[-1:][::-1][0]
                human_string = node_lookup.id_to_string(node_id)
                # score = predictions[node_id]
                self.save_image(adv_image, 'batch_adv ' + human_string)

                predictions = denoise_label
                predictions = np.squeeze(predictions)
                node_id = predictions.argsort()[-1:][::-1][0]
                human_string = node_lookup.id_to_string(node_id)
                # score = predictions[node_id]
                self.save_image(denoise_image, 'batch_denoise ' + human_string)

            counter += 1
            if counter % 100 == 0:
                precision_at_1 = count_top_1 / total_sample_count
                conf_at_1 = total_conf / total_sample_count
                recall_at_5 = count_top_5 / total_sample_count

                print("=== Step %d ===" % counter)
                print("top 1 accuracy: %.4f" % precision_at_1)
                print("top 1 confidence: %.4f" % conf_at_1)
                print("top 5 accuracy: %.4f" % recall_at_5)

                f.write("%d\t%.4f\t%.4f\t%.4f\t\n" % (counter, precision_at_1, conf_at_1, recall_at_5))

            if counter == val_num:
                coord.request_stop()
        coord.join(threads)

    def batch_adv_random_label_inference(self, val_num=1000, adv_name='fgsm', eps=0.007):
        """We expect a high accuracy when we calculate noise by fgsm with random label instead of correct label"""
        images, labels, synset, text, name, table = inputs(self.image_data, batch_size=1)

        top_1_op = tf.nn.in_top_k(self.y, labels, 1)
        top_5_op = tf.nn.in_top_k(self.y, labels, 5)
        table.init.run(session=self.sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        x_adv_tensor, eta = self.adversarize_batch_init(adv_name, eps)

        counter = 0
        count_top_1 = 0.0
        count_top_5 = 0.0
        total_correct_count = 0.0
        total_wrong_count = 0.0
        total_sample_count = 0.0
        total_conf = 0.0
        total_correct_conf = 0.0
        total_wrong_conf = 0.0

        while not coord.should_stop():
            images_data = self.sess.run([images,
                                         labels])

            label = np.random.randint(0, 1008, dtype='int32')
            while label == images_data[1][0]:
                label = np.random.randint(0, 1008, dtype='int32')

            adv_image, adv_label, noise = self.adversarize_batch(x_adv_tensor,
                                                                 eta,
                                                                 images_data[0],
                                                                 label,
                                                                 adv_name, eps)

            top_1 = self.sess.run(top_1_op, {self.rn_x: adv_image,
                                             labels: images_data[1]})
            top_5 = self.sess.run(top_5_op, {self.rn_x: adv_image,
                                             labels: images_data[1]})

            count_top_1 += np.sum(top_1)
            count_top_5 += np.sum(top_5)
            total_sample_count += 1

            predictions = self.sess.run(self.y, {self.rn_x: adv_image})
            predictions = np.squeeze(predictions)
            node_id = predictions.argsort()[-1:][::-1][0]
            score = predictions[node_id]
            total_conf += score
            if top_1:
                total_correct_count += 1
                total_correct_conf += score
            else:
                total_wrong_count += 1
                total_wrong_conf += score

            # node_lookup = icp.NodeLookup()
            # node_id = images_data[1][0]
            # human_string = node_lookup.id_to_string(node_id)
            # self.save_image(images_data[0], 'batch_nom ' + human_string)
            #
            # predictions = adv_label
            # predictions = np.squeeze(predictions)
            # node_id = predictions.argsort()[-1:][::-1][0]
            # human_string = node_lookup.id_to_string(node_id)
            # # score = predictions[node_id]
            # self.save_image(adv_image, 'batch_adv ' + human_string)

            counter += 1
            if counter % 100 == 0:
                precision_at_1 = count_top_1 / total_sample_count
                conf_at_1 = total_conf / total_sample_count
                recall_at_5 = count_top_5 / total_sample_count

                correct_conf_at_1 = total_correct_conf / total_correct_count
                wrong_conf_at_1 = total_wrong_conf / total_wrong_count

                print("=== Step %d ===" % counter)
                print("top 1 accuracy: %.4f" % precision_at_1)
                print("top 1 confidence: %.4f" % conf_at_1)
                print("top 5 accuracy: %.4f" % recall_at_5)

                print("top 1 correct confidence: %.4f" % correct_conf_at_1)
                print("top 1 wrong confidence: %.4f" % wrong_conf_at_1)

            if counter == val_num:
                coord.request_stop()
        coord.join(threads)

    def single_image_path(self, single_image_name):
        """
        Parameters
        ----------
        single_image_name: str, 'cropped_panda.jpg'
            For Inception network, which can only predict a single image, we need to provide a path for the image.
        """
        return os.path.join(util.ImageNet_DATA_DIR, "nom", "single_image", "%s.jpg" % single_image_name)

    def save_image(self, image_matrix, label):
        image_matrix = np.squeeze(image_matrix)
        save_path = os.path.join(util.ImageNet_DATA_DIR, "%s.jpg" % label)
        toimage(image_matrix).save(save_path)


if __name__ == '__main__':
    nva = ImageNetNvA()
    # must set the testing image first
    # nva.set_image_file('cropped_panda', 169)
    # nva.nom_test()
    # nva.adv_test('fgsm', 0.007)
    # nva.denoised_test()
    nva.set_batch_images()
    # nva.batch_inference(val_num=200)
    nva.batch_adv_inference(val_num=10000, adv_name='fgsm')
    # nva.batch_adv_random_label_inference(1000)
    # nva.batch_denoise_inference(val_num=50000,
    #                             adv_name='fgsm',
    #                             denoise_name='bilateral_cover')
