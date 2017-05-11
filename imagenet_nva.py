import os

import tensorflow as tf
import numpy as np
import adversarize
import denoise
import models.imagenet_inception as icp

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

    def batch_inference(self):
        pass

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
        adv_label : adarray
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
            denoise_label = self.sess.run(self.y, {self.rn_x: denoise_image})
        else:
            raise Exception("Unknown denoise method: %s \n" % denoise_name)
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
    nva.set_image_file('cropped_panda', 169)
    nva.nom_test()
    nva.adv_test('fgsm', 0.007)
    nva.denoised_test()
