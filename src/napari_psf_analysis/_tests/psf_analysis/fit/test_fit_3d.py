import numpy as np
from numpy.testing import assert_almost_equal

from napari_psf_analysis.psf_analysis.fit.fit_3d import evaluate_3d_gaussian
from napari_psf_analysis.psf_analysis.utils import estimate_from_data


def test_evaluate_3d_gaussian():
    zz, yy, xx = np.meshgrid(np.arange(50), np.arange(50), np.arange(50), indexing="ij")
    data = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], -1)

    bg, amp, mu_z, mu_y, mu_x, czz, czy, czx, cyy, cyx, cxx = (
        0,
        100,
        26,
        25,
        24,
        1,
        0,
        0.5,
        1,
        0.5,
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
    ).reshape(50, 50, 50)
    values = np.round(values).astype(int)

    est_bg, est_amp, est_mus, est_sigmas = estimate_from_data(values, values, (1, 1, 1))

    assert_almost_equal(est_bg, bg)
    assert_almost_equal(est_amp, amp)
    assert_almost_equal(est_mus[0], mu_z)
    assert_almost_equal(est_mus[1], mu_y)
    assert_almost_equal(est_mus[2], mu_x)
    assert_almost_equal(est_sigmas[0], 1, decimal=1)
    assert_almost_equal(est_sigmas[1], 1, decimal=1)
    assert_almost_equal(est_sigmas[2], 2, decimal=1)
