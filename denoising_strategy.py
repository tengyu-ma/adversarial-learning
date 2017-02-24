from numpy import *
from scipy.ndimage import filters


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


def threshold_method(im):
    # best performance until now
    # 0.7473		0.909817
    min_im = im.min()
    max_im = im.max()
    thres = 2 * min_im / (max_im - min_im) + 0.02
    im = (im - min_im) / (max_im - min_im)
    im[im < 0.38] = 0
    return im
