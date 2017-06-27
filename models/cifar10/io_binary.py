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


def compare_images_from_bin():
    IMAGE_SIZE = 24
    NUM_TO_GENERATE = 5
    ONE_LENGTH = 3 * IMAGE_SIZE * IMAGE_SIZE + 1
    file_name_org = os.path.join('/tmp/cifar10_data', 'cifar-10-batches-bin/test_batch_eps%d_org.bin' % EPS)
    file_name_noise = os.path.join('/tmp/cifar10_data', 'cifar-10-batches-bin/test_batch_eps%d_noise.bin' % EPS)
    f_org = open(file_name_org, 'rb')
    f_noise = open(file_name_noise, 'rb')
    file_org = f_org.read()
    file_noise = f_noise.read()
    file_org_list = list(file_org)
    file_noise_list = list(file_noise)
    total_num = int(len(file_org_list) / ONE_LENGTH)
    category = np.zeros(10)
    for i in range(NUM_TO_GENERATE):
        label_org = file_org_list[ONE_LENGTH * i]
        label_noise = file_noise_list[ONE_LENGTH * i]
        assert label_org == label_noise
        category[label_org] += 1
        # print(LABEL[label_org])

        # image_org = file_org_list[ONE_LENGTH * i + 1:ONE_LENGTH * (i + 1)]
        # image_org = np.reshape(image_org, (3, IMAGE_SIZE, IMAGE_SIZE))
        # image_org = np.transpose(image_org, (1, 2, 0)).astype('uint8')
        # image_noise = file_noise_list[ONE_LENGTH * i + 1:ONE_LENGTH * (i + 1)]
        # image_noise = np.reshape(image_noise, (3, IMAGE_SIZE, IMAGE_SIZE))
        # image_noise = np.transpose(image_noise, (1, 2, 0)).astype('uint8')
        #
        # fig = plt.figure()
        # plt.subplot(211)
        # plt.imshow(image_org)
        # plt.subplot(212)
        # plt.imshow(image_noise)
        # plt.show()

    print(category)
    f_org.close()
    f_noise.close()


def denoise_from_bin():
    pass


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
    # processing_images_without_noise()
    compare_images_from_bin()