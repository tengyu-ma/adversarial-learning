from numpy import *
from scipy.ndimage import filters
import cv2
import numpy as np


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


def bilateral_filter(im, epsilon):
    normlizer = 255 / (np.amax(im) - np.amin(im))
    im = (im + epsilon) * normlizer
    img = im.astype(np.uint8)

    # img_c = cv2.imread('panda1.jpg')
    img_output = cv2.bilateralFilter(img, 9, 20, 20)

    return img_output


def bm3d(im, epsilon):
    # some import error here
    import mvsfunc as mvf
    normlizer = 255 / (np.amax(im) - np.amin(im))
    im = (im + epsilon) * normlizer
    img = im.astype(np.uint8)

    img_output = mvf.BM3D(img, sigma=[3, 3, 3], radius1=0)  # radius1=0 for BM3D, radius1>0 for V-BM3D
    return img_output

# img = cv2.imread('panda1.jpg')
# contrast_enhancing(img)