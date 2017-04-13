import numpy as np
import matplotlib.pyplot as plt
import pickle

file_to_read = 'cifar-10-batches-py/test_batch.bin'
file_to_write = 'cifar-10-batches-py/test_batch.bin'
with open(file_to_read, 'rb') as f:
    file = f.read()
    file_list = list(file)
    for i in range(1):
        label = file_list[3073*i]
        image = file_list[3073*i+1:3073*(i+1)]
        image = list(map(float, image))
        image = np.reshape(image, (3, 32, 32))
        image_0 = np.transpose(image, (1, 2, 0))
        x = np.array([image_0[:,:,0], image_0[:,:,1], image_0[:,:,2]])
        image_1 = np.transpose(image, (2, 1, 0))
        # plt.imshow(image)
        # plt.show()