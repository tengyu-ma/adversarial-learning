import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from settings import *


def generate_pdf_files():
    IMAGE_SIZE = 24
    NUM_TO_GENERATE = 1
    ONE_LENGTH = 3 * IMAGE_SIZE * IMAGE_SIZE + 1

    file_name_org = os.path.join('/tmp/cifar10_data', 'cifar-10-batches-bin/test_batch_size24_eps%d_org.bin' % FLAGS.eps)
    f_org = open(file_name_org, 'rb')
    file_org = f_org.read()
    file_org_list = list(file_org)
    tmp = file_name_org.split('.')
    tmp[1] = 'pdf'
    file_name_org = '.'.join(tmp)
    # print(file_name_org)

    file_name_noise = os.path.join('/tmp/cifar10_data', 'cifar-10-batches-bin/test_batch_size24_eps%d_noise.bin' % FLAGS.eps)
    f_noise = open(file_name_noise, 'rb')
    file_noise = f_noise.read()
    file_noise_list = list(file_noise)

    file_name_denoised = os.path.join('/tmp/cifar10_data',
                                   'cifar-10-batches-bin/test_batch_size24_eps%d_denoised.bin' % FLAGS.eps)
    f_denoised = open(file_name_denoised, 'rb')
    file_denoised = f_denoised.read()
    file_denoised_list = list(file_denoised)

    for i in range(13,16):
        label_org = file_org_list[ONE_LENGTH * i]
        label_noise = file_noise_list[ONE_LENGTH * i]
        label_denoised = file_denoised_list[ONE_LENGTH * i]
        assert label_org == label_noise
        assert label_org == label_denoised
        print(LABEL[label_org])

        image_org = file_org_list[ONE_LENGTH * i + 1:ONE_LENGTH * (i + 1)]
        image_org = np.reshape(image_org, (3, IMAGE_SIZE, IMAGE_SIZE))
        image_org = np.transpose(image_org, (1, 2, 0)).astype('uint8')
        image_noise = file_noise_list[ONE_LENGTH * i + 1:ONE_LENGTH * (i + 1)]
        image_noise = np.reshape(image_noise, (3, IMAGE_SIZE, IMAGE_SIZE))
        image_noise = np.transpose(image_noise, (1, 2, 0)).astype('uint8')
        image_denoised = file_denoised_list[ONE_LENGTH * i + 1:ONE_LENGTH * (i + 1)]
        image_denoised = np.reshape(image_denoised, (3, IMAGE_SIZE, IMAGE_SIZE))
        image_denoised = np.transpose(image_denoised, (1, 2, 0)).astype('uint8')

        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(image_org)
        plt.subplot(132)
        plt.imshow(image_noise)
        plt.subplot(133)
        plt.imshow(image_denoised)
        plt.show()

    f_org.close()
    f_noise.close()


def compare_images_from_bin():
    IMAGE_SIZE = 24
    NUM_TO_GENERATE = 2
    ONE_LENGTH = 3 * IMAGE_SIZE * IMAGE_SIZE + 1
    file_name_org = os.path.join('/tmp/cifar10_data', 'cifar-10-batches-bin/test_batch_size24_eps%d_org.bin' % FLAGS.eps)
    file_name_noise = os.path.join('/tmp/cifar10_data', 'cifar-10-batches-bin/test_batch_size24_eps%d_noise.bin' % FLAGS.eps)
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
        print(LABEL[label_org])

        image_org = file_org_list[ONE_LENGTH * i + 1:ONE_LENGTH * (i + 1)]
        image_org = np.reshape(image_org, (3, IMAGE_SIZE, IMAGE_SIZE))
        image_org = np.transpose(image_org, (1, 2, 0)).astype('uint8')
        image_noise = file_noise_list[ONE_LENGTH * i + 1:ONE_LENGTH * (i + 1)]
        image_noise = np.reshape(image_noise, (3, IMAGE_SIZE, IMAGE_SIZE))
        image_noise = np.transpose(image_noise, (1, 2, 0)).astype('uint8')

        fig = plt.figure()
        plt.subplot(211)
        plt.imshow(image_org)
        plt.subplot(212)
        plt.imshow(image_noise)
        plt.show()

    print(category)
    f_org.close()
    f_noise.close()


def denoise_from_bin():
    IMAGE_SIZE = 32
    ONE_LENGTH = 3 * IMAGE_SIZE * IMAGE_SIZE + 1
    prefix = 'test_batch_eps%d' % FLAGS.eps
    file_name_noise = os.path.join('/tmp/cifar10_data/cifar-10-batches-bin', prefix+'_noise.bin')
    f_noise = open(file_name_noise, 'rb')
    file_noise = f_noise.read()
    file_noise_list = list(file_noise)
    total_num = int(len(file_noise_list) / ONE_LENGTH)
    for i in range(total_num):
        label_noise = file_noise_list[ONE_LENGTH * i]
        image_noise = file_noise_list[ONE_LENGTH * i + 1:ONE_LENGTH * (i + 1)]
        image_noise = np.reshape(image_noise, (3, IMAGE_SIZE, IMAGE_SIZE))
        image_noise = np.transpose(image_noise, (1, 2, 0)).astype('uint8')

        FLAGS.denoise_method = 'bilateral'
        if FLAGS.denoise_method == 'filter2D':
            kernel_org = [[0, 1, 0], [1, 4, 1], [0, 1, 0]]
            kernel = np.array(kernel_org) / np.sum(kernel_org) * 1.0
            image_denoised = cv2.filter2D(image_noise, -1, kernel)
        elif FLAGS.denoise_method == 'average':
            image_denoised = cv2.blur(image_noise, (2, 2))
        elif FLAGS.denoise_method == 'Gaussian':
            image_denoised = cv2.GaussianBlur(image_noise, (3, 3), 0)
        elif FLAGS.denoise_method == 'NLMeans':
            img = image_noise.astype(np.uint8)
            image_denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 7, 21)
        elif FLAGS.denoise_method == 'bilateral':
            image_denoised = cv2.bilateralFilter(image_noise, 9, 75, 75)
            image_denoised = cv2.bilateralFilter(image_denoised, 9, 75, 75)

        # fig = plt.figure()
        # plt.subplot(211)
        # plt.imshow(image_noise)
        # plt.subplot(212)
        # plt.imshow(image_denoised)
        # plt.show()

        image_matrix_denoised = np.array([image_denoised[:, :, 0], image_denoised[:, :, 1],
                                     image_denoised[:, :, 2]])
        image_list_denoised= np.array(image_matrix_denoised.flatten())
        image_list_denoised = list(image_list_denoised.astype('uint8'))

        label_list_denoised = list([label_noise])
        data_list_denoised = label_list_denoised + image_list_denoised
        data_bytes_denoised = bytes(data_list_denoised)

        if i == 0:
            file_denoised = data_bytes_denoised
        else:
            file_denoised += data_bytes_denoised

        if i % 200 == 0:
            print("Step: %d" % i)

    file_name_denoised = os.path.join('/tmp/cifar10_data/cifar-10-batches-bin', prefix+'_processed.bin')
    f_denoised = open(file_name_denoised, 'wb')
    f_denoised.write(file_denoised)
    f_denoised.close()


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
    # generate_pdf_files()
    # processing_images_without_noise()
    # compare_images_from_bin()
    denoise_from_bin()