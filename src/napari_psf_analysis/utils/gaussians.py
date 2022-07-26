import numpy as np


def gaussian_3d(
    height: float = 1,
    mu_z: float = 0,
    mu_y: float = 0,
    mu_x: float = 0,
    sigma_z: float = 2,
    sigma_y: float = 2,
    sigma_x: float = 2,
    offset: float = 0,
):
    """
    Return a parametrized 3D Gaussian function.

    Parameters:
        height: float
            Distance between the lowest and peak value of the Gaussian.
        mu_z: float
            Expected value of the Gaussian in Z dimension.
        mu_y: float
            Expected value of the Gaussian in Y dimension.
        mu_x: float
            Expected value of the Gaussian in X dimension.
        sigma_z: float
            Width of the Gaussian in Z dimension.
        sigma_y: float
            Width of the Gaussian in Y dimension.
        sigma_x: float
            Width of the Gaussian in X dimension.
        offset: float
            Shifts the Gaussian `up` or `down` i.e. the background signal.
    """
    return lambda z, y, x: offset + height * np.exp(
        -(
            ((z - mu_z) ** 2 / (2 * sigma_z ** 2))
            + ((y - mu_y) ** 2 / (2 * sigma_y ** 2))
            + ((x - mu_x) ** 2 / (2 * sigma_x ** 2))
        )
    )
