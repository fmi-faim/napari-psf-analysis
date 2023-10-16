import numpy as np
from numpy.testing import assert_almost_equal

from napari_psf_analysis.psf_analysis.fit.fit_1d import evaluate_1d_gaussian
from napari_psf_analysis.psf_analysis.utils import estimate_from_data


def test_evaluate_1d_gaussian():
    data = np.arange(50)

    bg, amp, mu, sigma = 10, 30, 22, 3

    values = evaluate_1d_gaussian(x=data, bg=bg, amp=amp, mu=mu, sigma=sigma)
    values = np.round(values).astype(int)

    est_bg, est_amp, est_mu, est_sigma = estimate_from_data(values, values, (1,))

    assert_almost_equal(est_bg, bg)
    assert_almost_equal(est_amp, amp)
    assert_almost_equal(est_mu, mu)
    assert_almost_equal(est_sigma, sigma, decimal=1)

    assert_almost_equal(values.max(), bg + amp, decimal=1)
    assert_almost_equal(np.argmax(values), mu)
