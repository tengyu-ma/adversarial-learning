import numpy as np
import matplotlib.pyplot as plt
import pickle

IMAGE_SIZE = 24
ONE_LENGTH = 3*IMAGE_SIZE*IMAGE_SIZE+1
file_to_read = 'C:\\tmp\\cifar10_data\\cifar-10-batches-bin\\test_batch_org_25.bin'
# file_to_write = 'cifar-10-batches-py/test_batch.bin'
with open(file_to_read, 'rb') as f:
    file = f.read()
    file_list = list(file)
    for i in range(5):
        label = file_list[ONE_LENGTH*i]
        image = file_list[ONE_LENGTH*i+1:ONE_LENGTH*(i+1)]
        # image = list(map(float, image))
        image = np.reshape(image, (3, IMAGE_SIZE, IMAGE_SIZE))
        image_0 = np.transpose(image, (1, 2, 0))
        # x = np.array([image_0[:,:,0], image_0[:,:,1], image_0[:,:,2]])
        # image_1 = np.transpose(image, (2, 1, 0))
        image_1 = image_0.astype('uint8')
        file_name = 'org_' + str(i) + '.pdf'

        # plt.imshow(image_1)
        # plt.savefig(file_name, format='pdf')
        fig = plt.figure()
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_axis_off()
        ax.imshow(image_1)
        fig.savefig(file_name)

        # plt.savefig(file_name, image_1)
        # plt.imshow(image_1)
        # plt.show()

        # fig = plt.figure(frameon=False)
        # fig.set_size_inches(0.24, 0.24)
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        # ax.imshow(image_1, aspect='auto')
        # plt.show()
        # fig.savefig('figure.png', dpi=1)
