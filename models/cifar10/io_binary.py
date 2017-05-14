import numpy as np
import matplotlib.pyplot as plt
import os
from settings import *


def generate_pdf_files():
    IMAGE_SIZE = 24
    ONE_LENGTH = 3 * IMAGE_SIZE * IMAGE_SIZE + 1
    file_to_read = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin/test_batch_eps%d_org.bin' % EPS)
    with open(file_to_read, 'rb') as f:
        file = f.read()
        file_list = list(file)
        for i in range(5):
            label = file_list[ONE_LENGTH * i]
            image = file_list[ONE_LENGTH * i + 1:ONE_LENGTH * (i + 1)]
            image = np.reshape(image, (3, IMAGE_SIZE, IMAGE_SIZE))
            image = np.transpose(image, (1, 2, 0)).astype('uint8')
            file_name = 'org_' + str(i) + '.pdf'

            fig = plt.figure()
            ax = fig.add_axes((0, 0, 1, 1))
            ax.set_axis_off()
            ax.imshow(image)
            # fig.savefig(file_name)
            plt.show()


def generate_images_without_noise():
    pass


if __name__ == '__main__':
    generate_images_without_noise()
