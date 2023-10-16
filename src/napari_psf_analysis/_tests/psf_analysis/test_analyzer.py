from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from napari_psf_analysis.psf_analysis.analyzer import Analyzer
from napari_psf_analysis.psf_analysis.fit.fit_3d import evaluate_3d_gaussian
from napari_psf_analysis.psf_analysis.parameters import PSFAnalysisInputs


@pytest.fixture
def img_data():
    zz, yy, xx = np.meshgrid(
        np.arange(150), np.arange(130), np.arange(140), indexing="ij"
    )
    data = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], -1)

    bg, amp, mu_z, mu_y, mu_x, czz, czy, czx, cyy, cyx, cxx = (
        250,
        5480,
        70,
        50,
        61,
        1,
        0,
        0,
        1,
        0,
        4,
    )

    return (
        evaluate_3d_gaussian(
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
        )
        .reshape(150, 130, 140)
        .astype(np.uint16)
    )


@pytest.fixture
def analyzer(img_data) -> Analyzer:
    return Analyzer(
        parameters=PSFAnalysisInputs(
            microscope="Test-Microscope",
            magnification=60,
            na=1.4,
            spacing=[200, 103.17, 103.17],
            patch_size=[6000, 2000, 2000],
            name="bead-image.tif",
            img_data=img_data,
            point_data=np.array([[68.2, 52.5, 61.7], [71.2, 52.2, 59.1]]),
            dpi=96,
            date="2023-10-16",
            version="1.0.2",
        )
    )


def test_iterator(analyzer: Analyzer):
    assert isinstance(analyzer, Iterator)
    assert len(analyzer) == 2


def test_progress(analyzer):
    for i, p in enumerate(analyzer):
        assert i + 1 == p


def test_results(analyzer: Analyzer):
    assert analyzer.get_results() is None

    for _ in enumerate(analyzer):
        pass

    result_table = analyzer.get_results()

    assert isinstance(result_table, pd.DataFrame)
    assert len(result_table) == 2

    cols = [
        "ImageName",
        "Date",
        "Microscope",
        "Magnification",
        "NA",
        "Amplitude_1D_Z",
        "Amplitude_2D_XY",
        "Amplitude_3D_XYZ",
        "Background_1D_Z",
        "Background_2D_XY",
        "Background_3D_XYZ",
        "Z_1D",
        "X_2D",
        "Y_2D",
        "X_3D",
        "Y_3D",
        "Z_3D",
        "FWHM_1D_Z",
        "FWHM_2D_X",
        "FWHM_2D_Y",
        "FWHM_3D_Z",
        "FWHM_3D_Y",
        "FWHM_3D_X",
        "FWHM_PA1_2D",
        "FWHM_PA2_2D",
        "FWHM_PA1_3D",
        "FWHM_PA2_3D",
        "FWHM_PA3_3D",
        "SignalToBG_1D_Z",
        "SignalToBG_2D_XY",
        "SignalToBG_3D_XYZ",
        "XYpixelsize",
        "Zspacing",
        "cov_xx_3D",
        "cov_xy_3D",
        "cov_xz_3D",
        "cov_yy_3D",
        "cov_yz_3D",
        "cov_zz_3D",
        "cov_xx_2D",
        "cov_xy_2D",
        "cov_yy_2D",
        "sde_amp_1D_Z",
        "sde_amp_2D_XY",
        "sde_amp_3D_XYZ",
        "sde_background_1D_Z",
        "sde_background_2D_XY",
        "sde_background_3D_XYZ",
        "sde_Z_1D",
        "sde_X_2D",
        "sde_Y_2D",
        "sde_X_3D",
        "sde_Y_3D",
        "sde_Z_3D",
        "sde_cov_xx_3D",
        "sde_cov_xy_3D",
        "sde_cov_xz_3D",
        "sde_cov_yy_3D",
        "sde_cov_yz_3D",
        "sde_cov_zz_3D",
        "sde_cov_xx_2D",
        "sde_cov_xy_2D",
        "sde_cov_yy_2D",
        "version",
    ]
    for column in result_table.columns:
        assert column in cols
        cols.remove(column)

    assert len(cols) == 0


def test_summary_figure_stack(analyzer: Analyzer):
    summaries, scaling = analyzer.get_summary_figure_stack(
        bead_img_scale=(200, 103.17, 103.17), bead_img_shape=(150, 130, 140)
    )
    assert summaries is None
    assert scaling is None

    for _ in analyzer:
        pass

    summaries, scaling = analyzer.get_summary_figure_stack(
        bead_img_scale=(200, 103.17, 103.17), bead_img_shape=(150, 130, 140)
    )

    assert summaries.shape == (2, 960, 960, 3)
    assert (scaling == (200, 13.9709375, 13.9709375)).all()
