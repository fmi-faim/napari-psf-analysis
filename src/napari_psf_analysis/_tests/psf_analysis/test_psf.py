import matplotlib
import numpy as np
import pytest

from napari_psf_analysis.psf_analysis.fit.fit_3d import evaluate_3d_gaussian
from napari_psf_analysis.psf_analysis.image import Calibrated3DImage
from napari_psf_analysis.psf_analysis.psf import PSF
from napari_psf_analysis.psf_analysis.records import PSFRecord

matplotlib.use("agg")


@pytest.fixture
def psf():
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
    return PSF(
        image=Calibrated3DImage(
            data=np.round(values).astype(int),
            spacing=(1, 1, 1),
        )
    )


def test_analyze(psf):
    psf.analyze()
    record = psf.get_record()

    assert isinstance(record, PSFRecord)
    record: PSFRecord = record

    assert record.z_fit is not None
    assert record.yx_fit is not None
    assert record.zyx_fit is not None

    assert isinstance(psf.get_summary_dict(), dict)


def test_summary_image(psf):
    psf.analyze()
    summary_fig = psf.get_summary_image(dpi=96, date="2023-10-16", version="1.0.0")
    assert isinstance(summary_fig, np.ndarray)
    assert summary_fig.shape == (960, 960, 3)


def test_summary_image_with_nan_in_z_fwhm(psf):
    psf.analyze()
    # Set FWHM to NaN
    psf.psf_record.z_fit.z_fwhm = np.nan
    summary_fig = psf.get_summary_image(dpi=96)
    assert isinstance(summary_fig, np.ndarray)
    assert summary_fig.shape == (960, 960, 3)


def test_summary_image_with_nan_in_y_fwhm(psf):
    psf.analyze()
    # Set FWHM to NaN
    psf.psf_record.yx_fit.y_fwhm = np.nan
    summary_fig = psf.get_summary_image(dpi=96)
    assert isinstance(summary_fig, np.ndarray)
    assert summary_fig.shape == (960, 960, 3)


def test_summary_image_with_nan_in_x_fwhm(psf):
    psf.analyze()
    # Set FWHM to NaN
    psf.psf_record.yx_fit.x_fwhm = np.nan
    summary_fig = psf.get_summary_image(dpi=96)
    assert isinstance(summary_fig, np.ndarray)
    assert summary_fig.shape == (960, 960, 3)
