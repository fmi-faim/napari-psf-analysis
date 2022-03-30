import datetime

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from napari_psf_analysis.psf_analysis.PSFAnalysis import PSFAnalysis
from napari_psf_analysis.utils.gaussians import gaussian_1d, gaussian_3d


@pytest.mark.parametrize(
    "mean, margin, start, end",
    [(23.4, 10, 13, 33), (23.5, 10, 14, 34), (23.6, 10, 14, 34)],
)
def test_create_slice(mean, margin, start, end):
    result = PSFAnalysis._create_slice(mean, margin)

    assert result == slice(start, end)


def test_guess_init_params():
    height = 4
    z, y, x = 5, 5, 4
    sigma_z, sigma_y, sigma_x = 2, 2, 2
    offset = 2

    Z, Y, X = np.meshgrid(range(10), range(10), range(10), indexing="ij")
    gaussian = gaussian_3d(height, z, y, x, sigma_z, sigma_y, sigma_x, offset)

    data = gaussian(Z, Y, X)

    result = PSFAnalysis._guess_init_params(data)

    assert_almost_equal(result[0], height, decimal=7)
    assert_almost_equal(result[1], z, decimal=0)
    assert_almost_equal(result[2], y, decimal=0)
    assert_almost_equal(result[3], x, decimal=0)
    assert_almost_equal(result[4], sigma_z, decimal=0)
    assert_almost_equal(result[5], sigma_y, decimal=0)
    assert_almost_equal(result[6], sigma_x, decimal=0)
    assert_almost_equal(result[7], offset, decimal=7)


def test_get_loss_function():
    height = 4
    z, y, x = 5, 5, 4
    sigma_z, sigma_y, sigma_x = 2, 2, 2
    offset = 2

    Z, Y, X = np.meshgrid(range(10), range(10), range(10), indexing="ij")
    gaussian = gaussian_3d(height, z, y, x, sigma_z, sigma_y, sigma_x, offset)

    data = gaussian(Z, Y, X)

    loss = PSFAnalysis._get_loss_function(data)

    result = loss((height, z, y, x, sigma_z, sigma_y, sigma_x, offset))

    assert_almost_equal(result, 0, decimal=7)


def test_fit_gaussian_3d():
    height = 4
    z, y, x = 5, 5, 4
    sigma_z, sigma_y, sigma_x = 2, 2, 2
    offset = 2

    Z, Y, X = np.meshgrid(range(10), range(10), range(10), indexing="ij")
    gaussian = gaussian_3d(height, z, y, x, sigma_z, sigma_y, sigma_x, offset)

    data = gaussian(Z, Y, X)

    result = PSFAnalysis._fit_gaussian_3d(data)

    assert_almost_equal(result[0], height, decimal=7)
    assert_almost_equal(result[1], z, decimal=7)
    assert_almost_equal(result[2], y, decimal=7)
    assert_almost_equal(result[3], x, decimal=7)
    assert_almost_equal(result[4], sigma_z, decimal=7)
    assert_almost_equal(result[5], sigma_y, decimal=7)
    assert_almost_equal(result[6], sigma_x, decimal=7)
    assert_almost_equal(result[7], offset, decimal=7)


def test_r_squared():
    height = 2
    mu = 4.5
    sigma = 2.1
    offset = 1

    gaussian = gaussian_1d(height, mu, sigma, offset)
    samples = gaussian(np.arange(10))

    result = PSFAnalysis._r_squared(samples, height, mu, sigma, offset)

    assert_almost_equal(result, 1, decimal=7)


def test_fwhm():
    sigma = 1.4

    fwhm = PSFAnalysis._fwhm(sigma)

    assert_almost_equal(fwhm / (2 * np.sqrt(2 * np.log(2))), sigma, decimal=7)


def test_get_signal():
    bead = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])

    result = PSFAnalysis._get_signal(bead, 0, 1, 1)

    assert_almost_equal(result, 1, decimal=7)


def test_psf_measure():
    height = 875
    z, y, x = 20, 20, 20
    sigma_z, sigma_y, sigma_x = 2, 1, 2
    offset = 150

    Z, Y, X = np.meshgrid(np.arange(40), np.arange(40), np.arange(40), indexing="ij")
    gaussian = gaussian_3d(height, z, y, x, sigma_z, sigma_y, sigma_x, offset)

    data = gaussian(Z, Y, X)

    psf_measure = PSFAnalysis(
        date=datetime.datetime(2022, 3, 29),
        microscope="test",
        magnification=1.0,
        NA=1.0,
        spacing=np.array([1, 1, 1]),
        patch_size=np.array([30, 30, 30]),
    )

    beads, offsets = psf_measure._localize_beads(data, np.array([[z, y, x]]))

    assert len(beads) == 1
    assert len(offsets) == 1
    assert beads[0].shape == tuple([30, 30, 30])
    assert_array_equal(beads[0], data[5:-5, 5:-5, 5:-5])
    assert_almost_equal(offsets[0], tuple([5, 5, 5]))

    beads, params, results = psf_measure.analyze(
        "test_img", data, np.array([[z, y, x]])
    )

    assert len(beads) == 1
    assert beads[0].shape == tuple([30, 30, 30])
    assert_array_equal(beads[0], data[5:-5, 5:-5, 5:-5])

    assert_almost_equal(params[0][0], height, decimal=7)
    assert_almost_equal(params[0][1], z - offsets[0][0], decimal=7)
    assert_almost_equal(params[0][2], y - offsets[0][1], decimal=7)
    assert_almost_equal(params[0][3], x - offsets[0][2], decimal=7)
    assert_almost_equal(params[0][4], sigma_z, decimal=7)
    assert_almost_equal(params[0][5], sigma_y, decimal=7)
    assert_almost_equal(params[0][6], sigma_x, decimal=7)
    assert_almost_equal(params[0][7], offset, decimal=7)

    assert results["ImageName"][0] == "test_img"
    assert results["Date"][0] == "2022-03-29"
    assert results["Microscope"][0] == "test"
    assert results["Magnification"][0] == 1.0
    assert results["NA"][0] == 1.0
    assert results["X"][0] == x
    assert results["Y"][0] == y
    assert results["Z"][0] == z
    assert_almost_equal(results["FWHM_X"][0], sigma_x * 2 * np.sqrt(2 * np.log(2)))
    assert_almost_equal(results["FWHM_Y"][0], sigma_y * 2 * np.sqrt(2 * np.log(2)))
    assert_almost_equal(results["FWHM_Z"][0], sigma_z * 2 * np.sqrt(2 * np.log(2)))
    assert_almost_equal(results["r2_x"][0], 1)
    assert_almost_equal(results["r2_x"][0], 1)
    assert_almost_equal(results["r2_x"][0], 1)
    assert_almost_equal(
        results["SignalToBG"][0], np.mean(data[20, 19:22, 19:22]) / offset
    )
    assert results["XYpixelsize"][0] == 1
    assert results["Zspacing"][0] == 1
