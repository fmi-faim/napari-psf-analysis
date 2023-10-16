import numpy as np
from numpy.testing import assert_almost_equal

from napari_psf_analysis.psf_analysis.fit.fit_2d import evaluate_2d_gaussian
from napari_psf_analysis.psf_analysis.utils import estimate_from_data


def test_evaluate_2d_gaussian():
    yy, xx = np.meshgrid(np.arange(50), np.arange(50), indexing="ij")
    data = np.stack([yy.ravel(), xx.ravel()], -1)

    bg, amp, mu_y, mu_x, cyy, cyx, cxx = 0, 100, 25, 24, 1, 0.5, 4

    values = evaluate_2d_gaussian(
        x=data, bg=bg, amp=amp, mu_y=mu_y, mu_x=mu_x, cyy=cyy, cyx=cyx, cxx=cxx
    ).reshape(50, 50)
    values = np.round(values).astype(int)

    est_bg, est_amp, est_mus, est_sigmas = estimate_from_data(values, values, (1, 1))

    assert_almost_equal(est_bg, bg)
    assert_almost_equal(est_amp, amp)
    assert_almost_equal(est_mus[0], mu_y)
    assert_almost_equal(est_mus[1], mu_x)
    assert_almost_equal(est_sigmas[0], 1, decimal=1)
    assert_almost_equal(est_sigmas[1], 2, decimal=1)
