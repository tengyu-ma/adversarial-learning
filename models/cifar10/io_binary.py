import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from settings import *


def generate_pdf_files():
    IMAGE_SIZE = 24
    NUM_TO_GENERATE = 5
    ONE_LENGTH = 3 * IMAGE_SIZE * IMAGE_SIZE + 1
    file_to_read = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin/test_batch_eps%d_org.bin' % EPS)
    with open(file_to_read, 'rb') as f:
        file = f.read()
        file_list = list(file)
        for i in range(NUM_TO_GENERATE):
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


def processing_images_without_noise():
    IMAGE_SIZE = 32
    ONE_LENGTH = 3 * IMAGE_SIZE * IMAGE_SIZE + 1
    for j in range(6):
        if j == 0:
            file_name = 'test_batch'
        else:
            file_name = 'data_batch_%d' % j
        file_to_read = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin/%s.bin' % file_name)
        with open(file_to_read, 'rb') as f:
            file = f.read()
            file_list = list(file)
            num_iter = int(len(file_list) / ONE_LENGTH)
            for i in range(num_iter):
                label = file_list[ONE_LENGTH * i]
                image = file_list[ONE_LENGTH * i + 1:ONE_LENGTH * (i + 1)]
                image = np.reshape(image, (3, IMAGE_SIZE, IMAGE_SIZE))
                image = np.transpose(image, (1, 2, 0)).astype('uint8')

                image_new = cv2.bilateralFilter(image, 9, 75, 75)
                image_matrix_new = np.array([image_new[:, :, 0], image_new[:, :, 1], image_new[:, :, 2]])
                image_list_new = np.array(image_matrix_new.flatten())
                image_list_new = list(image_list_new.astype('uint8'))
                data_list_new = [label] + image_list_new
                data_bytes_new = bytes(data_list_new)

                if i == 0:
                    file_new = data_bytes_new
                else:
                    file_new += data_bytes_new

                if i % 200 == 0:
                    print("%s, step: %d" % (file_name, i))

                # plt.figure(1)
                # plt.subplot(121)
                # plt.imshow(image)
                # plt.subplot(122)
                # plt.imshow(image_new)
                # plt.show()

            file_name_new = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin/%s_processed.bin' % file_name)
            with open(file_name_new, 'wb') as f:
                f.write(file_new)

if __name__ == '__main__':
    processing_images_without_noise()
