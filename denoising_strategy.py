from numpy import *
from scipy.ndimage import filters
import cv2


def threshold_method(im, thres=0.5):
    # best performance achieved when thres = 0.41, eps = 0.25, dilate with kernel (2,2)
    # 0.9394		0.975262
    min_im = im.min()
    max_im = im.max()
    im = (im - min_im) / (max_im - min_im)
    im[im <= thres] = 0
    im[im > thres] = 1

    kernel = ones((2, 2), uint8)

    # if isinstance(im[0][0],int):
    #     img = im.reshape(28, 28)
    #     tmp = cv2.dilate(img, kernel, iterations=1)
    #     im = tmp.flatten()
    # else:
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