import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from napari_psf_analysis.psf_analysis.fit.fit_3d import evaluate_3d_gaussian
from napari_psf_analysis.psf_analysis.fit.fitter import YXFitter, ZFitter, ZYXFitter
from napari_psf_analysis.psf_analysis.image import (
    Calibrated1DImage,
    Calibrated2DImage,
    Calibrated3DImage,
)
from napari_psf_analysis.psf_analysis.records import (
    YXFitRecord,
    ZFitRecord,
    ZYXFitRecord,
)
from napari_psf_analysis.psf_analysis.sample import YXSample, ZSample, ZYXSample
from napari_psf_analysis.psf_analysis.utils import fwhm

SPACING = (1, 1, 1)


@pytest.fixture
def sim_data():
    zz, yy, xx = np.meshgrid(np.arange(50), np.arange(30), np.arange(40), indexing="ij")
    data = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], -1)

    bg, amp, mu_z, mu_y, mu_x, czz, czy, czx, cyy, cyx, cxx = (
        250,
        5480,
        25,
        15,
        20,
        1,
        0,
        0,
        1,
        0,
        4,
    )

    values = evaluate_3d_gaussian(
        x=data,
        bg=bg,
        amp=amp,
        mu_z=mu_z,
        mu_y=mu_y,
        mu_x=mu_x,
        czz=czz,
        czy=czy,
        czx=czx,
        cyy=cyy,
        cyx=cyx,
        cxx=cxx,
    ).reshape(50, 30, 40)
    return Calibrated3DImage(
        data=np.round(values).astype(int),
        spacing=SPACING,
    )


@pytest.fixture
def z_fitter(sim_data):
    return ZFitter(image=sim_data)


@pytest.fixture
def yx_fitter(sim_data):
    return YXFitter(image=sim_data)


@pytest.fixture
def zyx_fitter(sim_data):
    return ZYXFitter(image=sim_data)


def test_get_z_sample(z_fitter, sim_data):
    sample = z_fitter._get_z_sample()

    assert isinstance(sample, ZSample)
    assert isinstance(sample.image, Calibrated1DImage)
    assert sample.image.offset == (0,)
    assert np.all(sample.image.data == sim_data.data[:, 15, 20])


def test_z_fit(z_fitter):
    params = z_fitter.fit()
    assert isinstance(params, ZFitRecord)
    params: ZFitRecord = params

    assert_almost_equal(params.z_bg, 250, decimal=2)
    assert_almost_equal(params.z_amp, 5480, decimal=1)
    assert_almost_equal(params.z_mu, 25 * SPACING[0], decimal=4)
    assert_almost_equal(params.z_sigma, 1, decimal=1)
    assert_almost_equal(params.z_fwhm, fwhm(1), decimal=1)
    assert params.z_bg_sde > 0
    assert params.z_amp_sde > 0
    assert params.z_mu_sde > 0
    assert params.z_sigma_sde > 0


def test_get_yx_sample(yx_fitter, sim_data):
    sample = yx_fitter._get_yx_sample()

    assert isinstance(sample, YXSample)
    assert isinstance(sample.image, Calibrated2DImage)
    assert sample.image.offset == (0, 0)
    assert np.all(sample.image.data == sim_data.data[25])


def test_yx_fit(yx_fitter):
    params = yx_fitter.fit()
    assert isinstance(params, YXFitRecord)
    params: YXFitRecord = params

    assert_almost_equal(params.yx_bg, 250, decimal=2)
    assert_almost_equal(params.yx_amp, 5480, decimal=0)
    assert_almost_equal(params.y_mu, 15 * SPACING[1], decimal=4)
    assert_almost_equal(params.x_mu, 20 * SPACING[2], decimal=4)
    assert_almost_equal(params.yx_cyy, 1, decimal=4)
    assert_almost_equal(params.yx_cyx, 0, decimal=4)
    assert_almost_equal(params.yx_cxx, 4, decimal=0)
    assert_almost_equal(params.y_fwhm, fwhm(1), decimal=0)
    assert_almost_equal(params.x_fwhm, fwhm(2), decimal=0)
    assert_almost_equal(params.yx_pc1_fwhm, fwhm(2), decimal=0)
    assert_almost_equal(params.yx_pc2_fwhm, fwhm(1), decimal=0)
    assert params.yx_bg_sde > 0
    assert params.yx_amp_sde > 0
    assert params.y_mu_sde > 0
    assert params.x_mu_sde > 0
    assert params.yx_cyy_sde > 0
    assert params.yx_cyx_sde > 0
    assert params.yx_cxx_sde > 0


def test_get_zyx_sample(zyx_fitter, sim_data):
    sample = zyx_fitter._estimator.sample

    assert isinstance(sample, ZYXSample)
    assert isinstance(sample.image, Calibrated3DImage)
    assert sample.image.offset == (0, 0, 0)
    assert np.all(sample.image.data == sim_data.data)


def test_zyx_fit(zyx_fitter):
    params = zyx_fitter.fit()
    assert isinstance(params, ZYXFitRecord)
    params: ZYXFitRecord = params

    assert_almost_equal(params.zyx_bg, 250, decimal=2)
    assert_almost_equal(params.zyx_amp, 5480, decimal=0)
    assert_almost_equal(params.zyx_z_mu, 25 * SPACING[0], decimal=4)
    assert_almost_equal(params.zyx_y_mu, 15 * SPACING[1], decimal=4)
    assert_almost_equal(params.zyx_x_mu, 20 * SPACING[2], decimal=4)
    assert_almost_equal(params.zyx_czz, 1, decimal=4)
    assert_almost_equal(params.zyx_czy, 0, decimal=4)
    assert_almost_equal(params.zyx_czx, 0, decimal=4)
    assert_almost_equal(params.zyx_cyy, 1, decimal=4)
    assert_almost_equal(params.zyx_cyx, 0, decimal=4)
    assert_almost_equal(params.zyx_cxx, 4, decimal=0)
    assert_almost_equal(params.zyx_z_fwhm, fwhm(1), decimal=0)
    assert_almost_equal(params.zyx_y_fwhm, fwhm(1), decimal=0)
    assert_almost_equal(params.zyx_x_fwhm, fwhm(2), decimal=0)
    assert_almost_equal(params.zyx_pc1_fwhm, fwhm(2), decimal=0)
    assert_almost_equal(params.zyx_pc2_fwhm, fwhm(1), decimal=0)
    assert_almost_equal(params.zyx_pc3_fwhm, fwhm(1), decimal=0)
    assert params.zyx_bg_sde > 0
    assert params.zyx_amp_sde > 0
    assert params.zyx_z_mu_sde > 0
    assert params.zyx_y_mu_sde > 0
    assert params.zyx_z_mu_sde > 0
    assert params.zyx_czz_sde > 0
    assert params.zyx_czy_sde > 0
    assert params.zyx_czx_sde > 0
    assert params.zyx_cyy_sde > 0
    assert params.zyx_cyx_sde > 0
    assert params.zyx_cxx_sde > 0
