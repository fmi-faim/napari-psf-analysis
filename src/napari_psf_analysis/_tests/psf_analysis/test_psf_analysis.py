import datetime

import numpy as np
import pkg_resources
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from napari_psf_analysis.psf_analysis.PSFAnalysis import PSFAnalysis
from napari_psf_analysis.utils.gaussians import gaussian_3d


@pytest.mark.parametrize(
    "mean, margin, start, end",
    [(23.4, 10, 13, 33), (23.5, 10, 14, 34), (23.6, 10, 14, 34)],
)
def test_create_slice(mean, margin, start, end):
    result = PSFAnalysis._create_slice(mean, margin)

    assert result == slice(start, end)


def test_fit_gaussian_3d():
    height = 4
    z, y, x = 5, 5, 4
    sigma_z, sigma_y, sigma_x = 2, 2, 2
    offset = 2

    Z, Y, X = np.meshgrid(range(10), range(10), range(10), indexing="ij")
    gaussian = gaussian_3d(height, z, y, x, sigma_z, sigma_y, sigma_x, offset)

    data = gaussian(Z, Y, X)

    result, _ = PSFAnalysis._fit_gaussian_3d(data, (1, 1, 1))

    assert_almost_equal(result[0], height, decimal=7)
    assert_almost_equal(result[1], offset, decimal=7)
    assert_almost_equal(result[2], x, decimal=7)
    assert_almost_equal(result[3], y, decimal=7)
    assert_almost_equal(result[4], z, decimal=7)
    assert_almost_equal(np.sqrt(result[5]), sigma_x, decimal=6)
    assert_almost_equal(np.sqrt(result[8]), sigma_y, decimal=6)
    assert_almost_equal(np.sqrt(result[10]), sigma_z, decimal=6)


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
    z, y, x = 20.5, 20.5, 20.5
    sigma_z, sigma_y, sigma_x = 5, 2, 1
    offset = 150

    Z, Y, X = np.meshgrid(np.arange(40), np.arange(40), np.arange(40), indexing="ij")
    gaussian = gaussian_3d(height, z, y, x, sigma_z, sigma_y, sigma_x, offset)

    data = gaussian(Z, Y, X)

    psf_measure = PSFAnalysis(
        date=datetime.datetime(2022, 3, 29),
        microscope="test",
        magnification=1,
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

    (beads, popts, pcovs, popts_2d, pcovs_2d, results) = psf_measure.analyze(
        "test_img", data, np.array([[z, y, x]])
    )

    assert len(beads) == 1
    assert beads[0].shape == tuple([30, 30, 30])
    assert_array_equal(beads[0], data[5:-5, 5:-5, 5:-5])

    result_height = popts[0][0]
    result_background = popts[0][1]
    result_mu_x = popts[0][2]
    result_mu_y = popts[0][3]
    result_mu_z = popts[0][4]
    cxx = popts[0][5]
    cxy = popts[0][6]
    cxz = popts[0][7]
    cyy = popts[0][8]
    cyz = popts[0][9]
    czz = popts[0][10]
    cov = np.array([[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]])
    eigval, eigvec = np.linalg.eig(cov)
    pa3, pa2, pa1 = np.sort(np.sqrt(eigval))

    result_height_2d = popts_2d[0][0]
    result_background_2d = popts_2d[0][1]
    result_mu_x_2d = popts_2d[0][2]
    result_mu_y_2d = popts_2d[0][3]
    cxx_2d = popts_2d[0][4]
    cxy_2d = popts_2d[0][5]
    cyy_2d = popts_2d[0][6]
    cov_2d = np.array([[cxx_2d, cxy_2d], [cxy_2d, cyy_2d]])
    eigval_2d, eigvec_2d = np.linalg.eig(cov_2d)
    pa2_2d, pa1_2d = np.sort(np.sqrt(eigval_2d))

    assert_almost_equal(result_height, height, decimal=0)
    assert_almost_equal(result_background, offset, decimal=0)
    assert_almost_equal(result_mu_z, z - offsets[0][0], decimal=7)
    assert_almost_equal(result_mu_y, y - offsets[0][1], decimal=7)
    assert_almost_equal(result_mu_x, x - offsets[0][2], decimal=7)
    assert_almost_equal(pa1, sigma_z, decimal=6)
    assert_almost_equal(pa2, sigma_y, decimal=6)
    assert_almost_equal(pa3, sigma_x, decimal=6)

    # This will not be exactly the simulated numbers, because the fit is
    # performed on the int(np.round(mu_z)) bead-plane even if the bead is
    # at a z sub-pixel location.
    assert_almost_equal(result_height_2d, 870, decimal=0)
    assert_almost_equal(result_background_2d, offset, decimal=0)
    assert_almost_equal(result_mu_y_2d, y - offsets[0][1], decimal=7)
    assert_almost_equal(result_mu_x_2d, x - offsets[0][2], decimal=7)
    assert_almost_equal(pa1_2d, sigma_y, decimal=7)
    assert_almost_equal(pa2_2d, sigma_x, decimal=7)

    assert results["ImageName"][0] == "test_img"
    assert results["Date"][0] == "2022-03-29"
    assert results["Microscope"][0] == "test"
    assert results["Magnification"][0] == 1.0
    assert results["NA"][0] == 1.0
    assert_almost_equal(results["X"][0], x)
    assert_almost_equal(results["Y"][0], y)
    assert_almost_equal(results["Z"][0], z)
    assert_almost_equal(results["X_2D"][0], x)
    assert_almost_equal(results["Y_2D"][0], y)
    assert_almost_equal(
        results["FWHM_X"][0], sigma_x * 2 * np.sqrt(2 * np.log(2)), decimal=6
    )
    assert_almost_equal(
        results["FWHM_Y"][0], sigma_y * 2 * np.sqrt(2 * np.log(2)), decimal=6
    )
    assert_almost_equal(
        results["FWHM_Z"][0], sigma_z * 2 * np.sqrt(2 * np.log(2)), decimal=6
    )
    fwhm_x = sigma_x * 2 * np.sqrt(2 * np.log(2))
    assert_almost_equal(results["FWHM_X_2D"][0], fwhm_x)
    fwhm_y = sigma_y * 2 * np.sqrt(2 * np.log(2))
    assert_almost_equal(results["FWHM_Y_2D"][0], fwhm_y)
    assert_almost_equal(results["SignalToBG"][0], height / offset)
    assert_almost_equal(results["SignalToBG_2D"][0], height / offset, decimal=1)
    assert results["XYpixelsize"][0] == 1
    assert results["Zspacing"][0] == 1
    version = pkg_resources.get_distribution("napari_psf_analysis").version
    assert results["version"][0] == version
