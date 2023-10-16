import numpy as np
from numpy.testing import assert_almost_equal
from scipy.ndimage import gaussian_filter
from skimage.util import img_as_float32, img_as_uint

from napari_psf_analysis.psf_analysis.utils import estimate_from_data, fwhm, sigma


def test_estimate_from_data():
    img = np.zeros((51, 51), dtype=np.uint8)

    mu_y = 23
    mu_x = 25
    sigma = 4
    amp = 431
    bg = 42

    img[mu_y, mu_x] = 1
    img = gaussian_filter(img_as_float32(img), sigma=sigma)
    img = img / img.max()
    img = img * (amp / 65535)
    img = img_as_uint(img)

    img = img_as_uint(img) + bg

    est_bg, est_amp, est_mus, est_sigmas = estimate_from_data(img, img, (1, 1))

    assert_almost_equal(est_bg, bg, decimal=7)
    assert_almost_equal(est_amp, amp, decimal=7)
    assert_almost_equal(est_mus[0], mu_y, decimal=7)
    assert_almost_equal(est_mus[1], mu_x, decimal=7)
    assert_almost_equal(est_sigmas[0], sigma, decimal=2)
    assert_almost_equal(est_sigmas[1], sigma, decimal=2)


def test_fwhm():
    assert_almost_equal(fwhm(1), 2.3548, decimal=4)


def test_sigma2fwhm():
    assert_almost_equal(sigma(fwhm(2)), 2, decimal=4)
