import pandas as pd
import pprint
import tensorflow as tf
from dytb.inputs.predefined import Cifar10, Cifar100
from dytb.train import train
from dytb.models.predefined.VGG import VGG
from dytb.models.predefined.SingleLayerCAE import SingleLayerCAE
from dytb.models.predefined.StackedCAE import StackedCAE

def process_image_with_autoenccoder():
    # vgg = VGG()
    cae = SingleLayerCAE()
    cifar10 = Cifar10.Cifar10()

    device = '/cpu:0'
    with tf.device(device):
        info = train(
            model=cae,
            dataset=cifar10,
            hyperparameters={
                "epochs": 1,
                "batch_size": 50,
                "regularizations": {
                    "l2": 1e-6,
                    "augmentation": {
                        "name": "FlipLR",
                        "fn": tf.image.random_flip_left_right
                    }
                },
                "gd": {
                    "optimizer": tf.train.AdamOptimizer,
                    "args": {
                        "learning_rate": 1e-3,
                        "beta1": 0.9,
                        "beta2": 0.99,
                        "epsilon": 1e-8
                    }
                }
            },
            force_restart=True)

if __name__ == '__main__':
    process_image_with_autoenccoder()