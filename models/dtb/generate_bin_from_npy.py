import numpy as np
from settings import *
import matplotlib.pyplot as plt
from PIL import Image
import cv2

IMAGE_SIZE = 24
NUM_TO_GENERATE = 5
ONE_LENGTH = 3 * IMAGE_SIZE * IMAGE_SIZE + 1

def main():
    input_images = np.load("/tmp/input_images.npy")
    output_images = np.load("/tmp/output_images.npy")
    input_labels = np.load("/tmp/input_labels.npy")
    autoencoder_test_set = FLAGS.autoencoder_test_set.split('.')[0]
    file_path_0 = '/tmp/cifar10_data/cifar-10-batches-bin/'+autoencoder_test_set+'.bin'
    file_path_before_cae = '/tmp/cifar10_data/cifar-10-batches-bin/'+autoencoder_test_set+'_before_cae.bin'
    file_path_after_cae = '/tmp/cifar10_data/cifar-10-batches-bin/'+autoencoder_test_set+'_after_cae.bin'
    # meta = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    input_images = input_images * 127.5 + 127.5
    assert np.max(input_images) < 256
    assert np.min(input_images) >= 0
    input_images = input_images.astype('uint8')

    file_to_write = []
    for i in range(10000):
        one_image = input_images[i]
        images_to_write = np.array([one_image[:, :, 0], one_image[:, :, 1], one_image[:, :, 2]])
        images_to_write = list((np.array(images_to_write)).flatten())
        label_to_write = list(input_labels[i])
        data_to_write = label_to_write + images_to_write
        if i == 0:
            file_to_write = data_to_write
        else:
            file_to_write.extend(data_to_write)
        if i % 200 == 0:
            print("Output file %s_before_cae.bin: %d/10000" % (autoencoder_test_set, i))

    file_to_write = bytes(np.array(file_to_write).astype('uint8'))

    with open(file_path_before_cae, 'wb') as f:
        f.write(file_to_write)
        # file_to_compare = f.read()
        # assert file_to_compare == file_to_write

    output_images = output_images * 127.5 + 127.5
    assert np.max(output_images) < 256
    assert np.min(output_images) >= 0
    output_images = output_images.astype('uint8')

    file_to_write = []
    for i in range(10000):
        one_image = output_images[i]
        images_to_write = np.array([one_image[:, :, 0], one_image[:, :, 1], one_image[:, :, 2]])
        images_to_write = list((np.array(images_to_write)).flatten())
        label_to_write = list(input_labels[i])
        data_to_write = label_to_write + images_to_write
        if i == 0:
            file_to_write = data_to_write
        else:
            file_to_write.extend(data_to_write)
        if i % 200 == 0:
            print("Output file %s_after_cae.bin: %d/10000" % (autoencoder_test_set, i))

    file_to_write = bytes(np.array(file_to_write).astype('uint8'))

    with open(file_path_after_cae, 'wb') as f:
        f.write(file_to_write)

    # IMAGE COMPARING
    # with open(file_path_0, 'rb') as f:
    #     file_to_compare = list(f.read())
    #     for i in range(0, 5):
    #         plt.figure(1)
    #         plt.subplot(121)
    #         plt.imshow(input_images[i])
    #         image_cmp = file_to_compare[ONE_LENGTH * i + 1:ONE_LENGTH * (i + 1)]
    #         image_cmp = np.reshape(image_cmp, (3, IMAGE_SIZE, IMAGE_SIZE))
    #         image_cmp = np.transpose(image_cmp, (1, 2, 0)).astype('uint8')
    #         plt.subplot(122)
    #         plt.imshow(output_images[i])
    #         plt.show()

    # for i in range(0, 5):
    #     file_name = 'input_' + str(i) + '.png'
    #     cv2.imwrite(file_name, input_images[i])

    # for i in range(0,1):
    #     plt.figure()
    #     plt.subplot(231)
    #     plt.imshow(input_images[i*6])
    #     plt.subplot(232)
    #     plt.imshow(input_images[i*6 + 1])
    #     plt.subplot(233)
    #     plt.imshow(input_images[i*6 + 2])
    #     plt.subplot(234)
    #     plt.imshow(input_images[i*6 + 3])
    #     plt.subplot(235)
    #     plt.imshow(input_images[i*6 + 4])
    #     plt.subplot(236)
    #     plt.imshow(input_images[i*6 + 5])
    #     tmp = int(input_labels[i])
    #     plt.title(meta[tmp])
    #     plt.show()

if __name__ == '__main__':
    main()

