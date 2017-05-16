import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

meta = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

input_images = np.load("input_images.npy")
output_images = np.load("output_images.npy")
input_labels = np.load("input_labels.npy")

assert np.max(input_images) <= 1
assert np.min(input_images) >= -1

input_images = input_images * 127.5 + 127.5

assert np.max(input_images) < 256
assert np.min(input_images) >= 0

input_images = input_images.astype('uint8')

assert np.max(output_images) <= 1
assert np.min(output_images) >= -1

output_images = output_images * 127.5 + 127.5

assert np.max(output_images) < 256
assert np.min(output_images) >= 0

output_images = output_images.astype('uint8')

# for i in range(0,5):
#     im = Image.fromarray(input_images[i])
#     im.show()

for i in range(0, 5):
    file_name = 'input_' + str(i) + '.png'
    cv2.imwrite(file_name, input_images[i])

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

# file_to_write = []
# for i in range(10000):
#     one_image = output_images[i]
#     images_to_write = np.array([one_image[:, :, 0], one_image[:, :, 1], one_image[:, :, 2]])
#     images_to_write = list((np.array(images_to_write)).flatten())
#     label_to_write = list(input_labels[i])
#     data_to_write = label_to_write + images_to_write
#     if i == 0:
#         file_to_write = data_to_write
#     else:
#         file_to_write.extend(data_to_write)
#     print("Image %d" % i)
#
# file_to_write = np.array(file_to_write).astype('uint8')
#
# with open('test_batch_after_cae.bin', 'wb') as f:
#     f.write(file_to_write)
