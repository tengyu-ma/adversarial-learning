from numpy import *
from scipy.ndimage import filters
import cv2
import numpy as np
import tensorflow as tf
import os
from scipy.misc import toimage
from utils import util


def threshold_method(im, thres=0.5):
    # best performance achieved when thres = 0.41, eps = 0.25, dilate with kernel (2,2)
    # 0.9394		0.975262
    min_im = im.min()
    max_im = im.max()
    im = (im - min_im) / (max_im - min_im)
    im[im <= thres] = 0
    im[im > thres] = 1

    kernel = ones((2, 2), uint8)

    for i in range(len(im)):
        img = im[i].reshape(28, 28)
        # tmp = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # good with (1,2) kernel
        # tmp = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # not good
        # tmp = cv2.erode(img,kernel,iterations = 1)  # not good
        tmp = cv2.dilate(img, kernel, iterations=1)  # best with (2,2) kernel
        im[i] = tmp.flatten()

    # thres1 = 2 * (0 - min_im) / (max_im - min_im) + adj
    # thres2 = 1 / (max_im - min_im) - adj
    # thres2 = 1 - thres1
    # im = im.flatten()
    # for i in range(len(im)):
    #     if im[i] < thres1:
    #         im[i] = 0
    #     elif im[i] > thres2:
    #         im[i] = 1
    #     else:
    #         im[i] = (im[i] - 0.5) * (max_im - min_im) + 0.5
    # im = im.reshape((-1,784))
    return im


def bilateral_mnist_method(im, thres=0.5, d=5, sigma_color=10, sigma_space=10):
    # best performance achieved when thres = 0.41, eps = 0.25, dilate with kernel (2,2)
    # 0.9394		0.975262
    min_im = im.min()
    max_im = im.max()
    im = (im - min_im) / (max_im - min_im)
    im[im <= thres] = 0
    im[im > thres] = 1

    for i in range(len(im)):
        img = im[i].reshape(28, 28)
        tmp = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        im[i] = tmp.flatten()

    # thres1 = 2 * (0 - min_im) / (max_im - min_im) + adj
    # thres2 = 1 / (max_im - min_im) - adj
    # thres2 = 1 - thres1
    # im = im.flatten()
    # for i in range(len(im)):
    #     if im[i] < thres1:
    #         im[i] = 0
    #     elif im[i] > thres2:
    #         im[i] = 1
    #     else:
    #         im[i] = (im[i] - 0.5) * (max_im - min_im) + 0.5
    # im = im.reshape((-1,784))
    return im


def test_method(im, thres=0.5):
    # min_im = im.min()
    # max_im = im.max()
    # im = (im - min_im) / (max_im - min_im)
    #
    # im[im <= thres] = 0
    # im[im > thres] = 1

    return cv2.medianBlur(im, 3)
    #
    # kernel = ones((2, 2), uint8)

    # for i in range(len(im)):
    #     img = im[i].reshape(28, 28)
    #     tmp = cv2.dilate(img, kernel, iterations=1)  # best with (2,2) kernel
    #     im[i] = tmp.flatten()

    # return im


def rof(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    # much worse than the situation without denoising
    # without denoising: 0.1721		0.909802
    # with this denoising: 0.1204		0.979649
    m, n = im.shape

    U = U_init
    Px = im
    Py = im
    error = 1

    while error > tolerance:
        Uold = U

        GradUx = roll(U, -1, axis=1) - U
        GradUy = roll(U, -1, axis=0) - U

        PxNew = Px + (tau / tv_weight) * GradUx
        PyNew = Py + (tau / tv_weight) * GradUy
        NormNew = maximum(1, sqrt(PxNew ** 2 + PyNew ** 2))

        Px = PxNew / NormNew
        Py = PyNew / NormNew

        RxPx = roll(Px, 1, axis=1)
        RyPy = roll(Py, 1, axis=0)

        DivP = (Px - RxPx) + (Py - RyPy)
        U = im + tv_weight * DivP

        error = linalg.norm(U - Uold) / sqrt(n * m)

    return U


def gaussian(im):
    # a bit better, confidence much lower
    return filters.gaussian_filter(im, 1)


def contrast_enhancing(im, epsilon):
    """ does not work, even worse """
    normlizer = 255 / (np.amax(im) - np.amin(im))
    im = (im + epsilon) * normlizer
    img = im.astype(np.uint8)
    img_c = cv2.imread('panda1.jpg')

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # cv2.imshow('Color input image', img)
    # cv2.waitKey(0)
    #
    # cv2.imshow('Histogram equalized', img_output)
    # cv2.waitKey(0)

    return img_output


def fast_nl_means_denoising_colored(im, epsilon):
    normlizer = 255 / (np.amax(im) - np.amin(im))
    im = (im + epsilon) * normlizer
    img = im.astype(np.uint8)

    # img_c = cv2.imread('panda1.jpg')
    img_output = cv2.fastNlMeansDenoisingColored(img)

    return img_output


def bilateral_filter(im, d=9, sigma_color=20, sigma_space=20):
    """Bilateral filter
    
    Parameters
    ----------
    im : ndarray
        the input image data
    d : int, 9
        Diameter of each pixel neighborhood that is used during filtering.
        If it is non-positive, it is computed from sigmaSpace .
    sigma_color : int, 20
        Filter sigma in the color space. 
        A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace )
        will be mixed together, resulting in larger areas of semi-equal color.
    sigma_space : int, 20
        Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence
        each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood
        size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace .
        
    Returns
    -------
    image_output : ndarray
        the filtered output data

    """
    # normlizer = 255 / (np.amax(im) - np.amin(im))
    # im = (im + epsilon) * normlizer
    # img = im.astype(np.uint8)
    # img_c = cv2.imread('panda1.jpg')
    im = np.squeeze(im)
    img_output = cv2.bilateralFilter(im, d, sigma_color, sigma_space)
    # img_output = img_output * 2.0 + 0.014 # increase contrast
    img_output = np.expand_dims(img_output, 0)
    return img_output


def contrast(im):
    img_output = im + 0.007
    return img_output


def guassian_imagenet(im):
    img_output = filters.gaussian_filter(im, 1)
    return img_output


def erosion_imagenet(im):
    """Erosion filter

    Parameters
    ----------
    im : ndarray
        the input image data

    Returns
    -------
    image_output : ndarray
        the filtered output data

    """
    # normlizer = 255 / (np.amax(im) - np.amin(im))
    # im = (im + epsilon) * normlizer
    # img = im.astype(np.uint8)
    # img_c = cv2.imread('panda1.jpg')
    kernel = np.ones((5, 5), np.uint8)
    im = np.squeeze(im)
    img_output = cv2.erode(im, kernel, iterations=1)
    img_output = np.expand_dims(img_output, 0)
    return img_output


def dilation_imagenet(im):
    """Erosion filter

    Parameters
    ----------
    im : ndarray
        the input image data

    Returns
    -------
    image_output : ndarray
        the filtered output data

    """
    # normlizer = 255 / (np.amax(im) - np.amin(im))
    # im = (im + epsilon) * normlizer
    # img = im.astype(np.uint8)
    # img_c = cv2.imread('panda1.jpg')
    kernel = np.ones((5, 5), np.uint8)
    im = np.squeeze(im)
    img_output = cv2.dilate(im, kernel, iterations=1)
    img_output = np.expand_dims(img_output, 0)
    return img_output


def opening_imagenet(im):
    """Opening filter

    Parameters
    ----------
    im : ndarray
        the input image data

    Returns
    -------
    image_output : ndarray
        the filtered output data

    """
    # normlizer = 255 / (np.amax(im) - np.amin(im))
    # im = (im + epsilon) * normlizer
    # img = im.astype(np.uint8)
    # img_c = cv2.imread('panda1.jpg')
    kernel = np.ones((4, 4), np.uint8)
    im = np.squeeze(im)
    img_output = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    img_output = np.expand_dims(img_output, 0)
    return img_output


def closing_imagenet(im):
    """Closing filter

    Parameters
    ----------
    im : ndarray
        the input image data

    Returns
    -------
    image_output : ndarray
        the filtered output data

    """
    # normlizer = 255 / (np.amax(im) - np.amin(im))
    # im = (im + epsilon) * normlizer
    # img = im.astype(np.uint8)
    # img_c = cv2.imread('panda1.jpg')
    kernel = np.ones((5, 5), np.uint8)
    im = np.squeeze(im)
    img_output = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    img_output = np.expand_dims(img_output, 0)
    return img_output


def random_cover_imagenet(im, epsilon=0.007):
    im = np.squeeze(im)
    # im = float_to_uint8(im)

    # random_sign = np.random.randint(0, 2, size=(299, 299, 3))
    # random_sign[random_sign == 0] = -1
    # noise = random_sign * epsilon
    # noise = random_sign * 0.007

    laplacian = cv2.Laplacian(im, cv2.CV_32F)
    # noise = laplacian
    noise = np.sign(laplacian) * 0.007

    # gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
    # gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
    # mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # noise = mag / np.amax(im) * 0.007
    # noise = mag

    # save_image(gx, 'gx')
    # save_image(gy, 'gy')
    # save_image(mag, 'mag')
    # save_image(laplacian, 'laplacian')
    # save_image(noise, 'fgsm_sobel')

    img_output = im - noise
    # img_output = np.expand_dims(img_output, 0)
    return img_output


def bilateral_and_cover(im, d=9, sigma_color=15, sigma_space=15):
    """Bilateral filter

    Parameters
    ----------
    im : ndarray
        the input image data
    d : int, 9
        Diameter of each pixel neighborhood that is used during filtering.
        If it is non-positive, it is computed from sigmaSpace .
    sigma_color : int, 20
        Filter sigma in the color space. 
        A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace )
        will be mixed together, resulting in larger areas of semi-equal color.
    sigma_space : int, 20
        Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence
        each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood
        size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace .

    Returns
    -------
    image_output : ndarray
        the filtered output data

    """
    # normlizer = 255 / (np.amax(im) - np.amin(im))
    # im = (im + epsilon) * normlizer
    # img = im.astype(np.uint8)
    # img_c = cv2.imread('panda1.jpg')
    im = np.squeeze(im)

    laplacian = cv2.Laplacian(im, cv2.CV_32F)
    noise = np.sign(laplacian) * 0.007

    img_output = cv2.bilateralFilter(im, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    img_output = img_output - noise

    img_output = np.expand_dims(img_output, 0)
    return img_output


def counter_fgsm_imagenet_batch_init(J, rn_x, epsilon=2):
    """
    A fast gradient sign method to generate adversarial example

    Parameters
    ----------
    J : tensor
        cost function
    rn_x : tensor
        tensorflow tensor for the resized and normalized input image
    epsilon :
        noise rate

    Returns
    -------
    x_adv_tensor : 
        tensor for adversarial image
    eta : 
        the noise tensor added to the normal data; noise = sign(▽_xJ(theta,x,y))
    """
    nabla_J = tf.gradients(J, rn_x)  # apply nabla operator to calculate the gradient
    sign_nabla_J = tf.sign(nabla_J)  # calculate the sign of the gradient of cost function
    eta = tf.multiply(sign_nabla_J, epsilon)  # multiply epsilon the sign of the gradient of cost function
    # x_adv_tensor = tf.add(rn_x, tf.squeeze(eta))
    x_adv_tensor = tf.subtract(rn_x, tf.squeeze(eta))
    # noise = sess.run(eta, {rn_x: image_data, y_: image_label})
    # noise = None
    # x_adv = sess.run(tf.add(rn_x, tf.squeeze(eta)), {rn_x: image_data, y_: image_label})  # add noise to test data

    # eta = nabla_J
    return x_adv_tensor, eta


def counter_fgsm(sess, counter_adv_tensor, counter_eta,
                 J, x, y_, rn_x, image_data, image_label, epsilon=2):
    """Bilateral filter

    Parameters
    ----------
    sess : Session
        default tensorflow session
    counter_adv_tensor : tensor
        tensor for adversarial image
    counter_eta : tensor
        tensor for adversarial noise
    J : tensor
        cost function
    x : placeholder
        placeholder for the input image
    y_ : placeholder
        placeholder for the correct one-hot label
    rn_x : tensor
        tensorflow tensor for the resized and normalized input image
    image_data : ndarray
        resized and normalized input data
    image_label : ndarray
        one-hot correct label
    epsilon :
        noise rate
        
    Returns
    -------
    image_output : ndarray
        the filtered output data

    """
    # normlizer = 255 / (np.amax(im) - np.amin(im))
    # im = (im + epsilon) * normlizer
    # img = im.astype(np.uint8)
    # img_c = cv2.imread('panda1.jpg')
    noise = sess.run(counter_eta, {rn_x: image_data, y_: image_label})
    x_adv = sess.run(counter_adv_tensor, {rn_x: image_data, y_: image_label})

    return x_adv, noise


def gradient_imagenet(im):
    """Gradient filter

    Parameters
    ----------
    im : ndarray
        the input image data

    Returns
    -------
    image_output : ndarray
        the filtered output data

    """
    # normlizer = 255 / (np.amax(im) - np.amin(im))
    # im = (im + epsilon) * normlizer
    # img = im.astype(np.uint8)
    # img_c = cv2.imread('panda1.jpg')
    kernel = np.ones((5, 5), np.uint8)
    im = np.squeeze(im)
    img_output = cv2.morphologyEx(im, cv2.MORPH_GRADIENT, kernel)
    img_output = np.expand_dims(img_output, 0)
    return img_output


def equalizeHist_imagenet(im):
    """equalizeHist filter.
    Only work for grayscale image

    Parameters
    ----------
    im : ndarray
        the input image data

    Returns
    -------
    image_output : ndarray
        the filtered output data

    """

    im = np.squeeze(im)
    img_output = cv2.equalizeHist(im)
    img_output = np.expand_dims(img_output, 0)
    return img_output


def bm3d(im, epsilon):
    # some import error here
    import utils.mvsfunc as mvf
    normlizer = 255 / (np.amax(im) - np.amin(im))
    im = (im + epsilon) * normlizer
    img = im.astype(np.uint8)

    img_output = mvf.BM3D(img, sigma=[3, 3, 3], radius1=0)  # radius1=0 for BM3D, radius1>0 for V-BM3D
    return img_output


def float_to_uint8(im):
    normlizer = 255 / (np.amax(im) - np.amin(im))
    im = im * normlizer
    im = im.astype(np.uint8)
    return im


def save_image(image_matrix, label):
    image_matrix = np.squeeze(image_matrix)
    save_path = os.path.join(util.ImageNet_DATA_DIR, "%s.jpg" % label)
    toimage(image_matrix).save(save_path)
