"""
This file is designed for integrating different DNNs, since some are developed by ourselves, some are developed by
third party which may has already been trained.
"""

from normal_vs_adversarial import NormalVsAdversarial
from util import SUPPORTED_DNNS
import inception_ImageNet as icp_IN


class NomVsAdv:
    def __init__(self, network_name, trained=True):
        """
        Parameters
        ----------
        network_name : str
            The name of network we are going to use.
        trained : boolean
            The initial network is trained or not
            
        Attributes
        ----------
        DNNs: class
            the deep neural network from different source
        trained : boolean
            The initial network is trained or not
        """
        self.DNNs = None
        self.DNNs_name = network_name
        self.trained = trained
        self.set_DNNs(network_name)

    def set_DNNs(self, network_name, train_iter=1000):
        assert network_name in SUPPORTED_DNNS, "%s is not a supported network" % network_name

        if network_name == 'Inception':
            self.trained = True  # inception is a already trained network
            self.DNNs = 'Real network is stored in inception_ImageNet.py'
        elif network_name == 'CIFAR-10':
            # self.DNNs = ?
            print('left for Huahong')
        else:
            self.DNNs = NormalVsAdversarial()
            if self.trained:
                self.DNNs.restore_network(network_name)
            else:
                self.DNNs.set_iter(train_iter)
                self.DNNs.run_training(network_name)

    def set_single_image(self, single_iamge_name):
        """
        Parameters
        ----------
        single_iamge_name: str, 'cropped_panda.jpg'
            For Inception network, which can only predict a single image, we need to provide a path for the image.
        """
        if self.DNNs_name == 'Inception':
            icp_IN.FLAGS.image_file = single_iamge_name

    def nom_test(self):
        if self.DNNs_name == 'Inception':
            icp_IN.make_prediction()
        elif self.DNNs_name == 'CIFAR-10':
            print("left for Huahong")
        else:
            self.DNNs.normal_test()
            self.DNNs.sess.close()

    def adv_test(self):
        # ToDo: add adversarial noise to ImageNet and CIFAR-10
        if self.DNNs_name == 'Inception':
            pass

    def denoised_test(self):
        # ToDo: add denoise to ImageNet and CIFAR-10
        if self.DNNs_name == 'Inception':
            pass


if __name__ == '__main__':
    """ our own network for MNIST"""
    nva = NomVsAdv('ReLU_Softmax_AdamOptimizer')
    """ Inception network for ImageNet"""
    # nva = NomVsAdv('Inception')
    # nva.set_single_image('grace_hopper.jpg')  # default is cropped_panda.jpg
    """ Huahong's network for CIFAR-10"""

    nva.nom_test()


