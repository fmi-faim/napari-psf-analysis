import numpy as np
from numpy.testing import assert_almost_equal

from napari_psf_analysis.psf_analysis.fit_3d import evaluate_3d_gaussian
from napari_psf_analysis.psf_analysis.psf_analysis import (
    analyze_bead,
    localize_beads,
    merge,
)


def test_localize_beads():
    img = np.zeros((50, 50, 50), dtype=np.uint8)

    img = img + np.random.normal(50, 4, img.shape)
    img[20, 21, 20] = 200
    img[23, 24, 27] = 500

    crops, offsets = localize_beads(
        img=img,
        points=np.array([[21, 22, 26]]),
        patch_size=(21, 21, 21),
        spacing=(1, 1, 1),
    )
    bead = crops[0]
    assert bead.shape == (21, 21, 21)
    assert_almost_equal(bead[10, 10, 10], 500)


def test_merge():
    accumulated_results = None
    results = {"a": 1, "b": 2}

    accumulated_results = merge(accumulated_results=accumulated_results, result=results)

    assert accumulated_results == {"a": [1], "b": [2]}

    accumulated_results = merge(accumulated_results=accumulated_results, result=results)

    assert accumulated_results == {"a": [1, 1], "b": [2, 2]}


def test_analyze_bead():
    z, y, x = np.meshgrid(
        np.arange(128) * 0.200,
        np.arange(256) * 0.134,
        np.arange(256) * 0.134,
        indexing="ij",
    )

    bead_img = (
        evaluate_3d_gaussian(
            x=np.stack([z.ravel(), y.ravel(), x.ravel()], -1),
            bg=40,
            amp=320,
            mu_z=12.0,
            mu_y=17.152,
            mu_x=16.942,
            czz=4,
            czy=0,
            czx=0,
            cyy=1,
            cyx=0,
            cxx=1.1**2,
        )
        .reshape(128, 256, 256)
        .astype(np.uint16)
    )

    spacing = (200, 134, 134)
    psfs, offsets = localize_beads(
        img=bead_img,
        points=[[64, 128, 128]],
        patch_size=(12000, 8000, 8000),
        spacing=spacing,
    )

    res = analyze_bead(
        bead=psfs[0],
        spacing=spacing,
    )

    assert_almost_equal(res["z_bg"], 40, decimal=0)
    assert_almost_equal(res["z_amp"], 320, decimal=0)
    assert_almost_equal(res["z_mu"], 6000, decimal=0)
    assert_almost_equal(res["z_sigma"], 2000, decimal=0)
    assert_almost_equal(res["yx_bg"], 40, decimal=0)
    assert_almost_equal(res["yx_amp"], 320, decimal=0)
    assert_almost_equal(res["y_mu"] + offsets[0][1] * 134, 17152, decimal=0)
    assert_almost_equal(res["x_mu"] + offsets[0][2] * 134, 16942, decimal=0)
    assert_almost_equal(np.sqrt(res["yx_cyy"]), 1000, decimal=0)
    assert_almost_equal(np.sqrt(res["yx_cxx"]), 1100, decimal=0)
    assert_almost_equal(res["z_fwhm"], 4710, decimal=0)
    assert_almost_equal(res["y_fwhm"], 2352, decimal=0)
    assert_almost_equal(res["x_fwhm"], 2587, decimal=0)
    assert_almost_equal(res["zyx_bg"], 40, decimal=0)
    assert_almost_equal(res["zyx_amp"], 320, decimal=0)
    assert_almost_equal(res["zyx_z_mu"], 6000, decimal=0)
    assert_almost_equal(res["zyx_y_mu"] + offsets[0][1] * 134, 17152, decimal=0)
    assert_almost_equal(res["zyx_x_mu"] + offsets[0][2] * 134, 16942, decimal=0)
