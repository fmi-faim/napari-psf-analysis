import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from napari_psf_analysis.utils.gaussians import gaussian_3d


@pytest.mark.parametrize(
    "height, mu_z, mu_y, mu_x, sigma_z, sigma_y, sigma_x, offset",
    [(1, 5, 5, 5, 2, 2, 2, 0), (1, 5, 5, 5, 2, 2, 2, 2), (1, 4, 5, 5, 1, 2, 2, 0)],
)
def test_gaussian_3d(height, mu_z, mu_y, mu_x, sigma_z, sigma_y, sigma_x, offset):
    z = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    x = np.linspace(0, 10, 100)

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    gaussian = gaussian_3d(
        height=height,
        mu_z=mu_z,
        mu_y=mu_y,
        mu_x=mu_x,
        sigma_z=sigma_z,
        sigma_y=sigma_y,
        sigma_x=sigma_x,
        offset=offset,
    )

    g = gaussian(Z, Y, X)

    assert_almost_equal(Z.ravel()[np.argmax(g)], mu_z, decimal=1)
    assert_almost_equal(Y.ravel()[np.argmax(g)], mu_y, decimal=1)
    assert_almost_equal(X.ravel()[np.argmax(g)], mu_x, decimal=1)
    assert_almost_equal(g.min(), offset, decimal=2)
    assert_almost_equal(g.max() - offset, height, decimal=2)

    fwhm = (
        Z[::-1, mu_y * 10, mu_x * 10][
            np.argmax(g[::-1, mu_y * 10, mu_x * 10] > (offset + height / 2))
        ]
        - Z[
            np.argmax(g[:, mu_y * 10, mu_x * 10] > (offset + height / 2)),
            mu_y * 10,
            mu_x * 10,
        ]
    )
    assert_almost_equal(fwhm / (2 * np.sqrt(2 * np.log(2))), sigma_z, decimal=1)
    fwhm = (
        Y[mu_z * 10, ::-1, mu_x * 10][
            np.argmax(g[mu_z * 10, ::-1, mu_x * 10] > (offset + height / 2))
        ]
        - Y[
            mu_z * 10,
            np.argmax(g[mu_z * 10, :, mu_x * 10] > (offset + height / 2)),
            mu_x * 10,
        ]
    )
    assert_almost_equal(fwhm / (2 * np.sqrt(2 * np.log(2))), sigma_y, decimal=1)
    fwhm = (
        X[mu_z * 10, mu_y * 10, ::-1][
            np.argmax(g[mu_z * 10, mu_y * 10, ::-1] > (offset + height / 2))
        ]
        - X[
            mu_z * 10,
            mu_y * 10,
            np.argmax(g[mu_z * 10, mu_y * 10, :] > (offset + height / 2)),
        ]
    )
    assert_almost_equal(fwhm / (2 * np.sqrt(2 * np.log(2))), sigma_x, decimal=1)
