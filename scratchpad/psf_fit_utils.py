import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.special import kl_div
from skimage.measure import centroid


def get_raw_crop(img, size_nm, spacing):
    z, y, x = np.unravel_index(np.argmax(img), img.shape)

    half_size = np.ceil(size_nm / spacing / 2).astype(int)

    z_slice = slice(max(0, z - half_size[0]), min(z + half_size[0] + 1, img.shape[0]))
    y_slice = slice(max(0, y - half_size[1]), min(y + half_size[1] + 1, img.shape[1]))
    x_slice = slice(max(0, x - half_size[2]), min(x + half_size[2] + 1, img.shape[2]))

    crop = img[z_slice, y_slice, x_slice]
    return crop


def get_cov_matrix(img, spacing):
    def cov(x, y, i):
        return np.sum(x * y * i) / np.sum(i)

    extends = [np.arange(dim_size) * s for dim_size, s in zip(img.shape, spacing)]

    grids = np.meshgrid(
        *extends,
        indexing="ij",
    )
    cen = centroid(img)

    if img.ndim == 1:
        x = grids[0].ravel() - cen[0] * spacing[0]
        return cov(x, x, img.ravel())
    elif img.ndim == 2:
        y = grids[0].ravel() - cen[0] * spacing[0]
        x = grids[1].ravel() - cen[1] * spacing[1]

        cxx = cov(x, x, img.ravel())
        cyy = cov(y, y, img.ravel())
        cxy = cov(x, y, img.ravel())

        return np.array([[cxx, cxy], [cxy, cyy]])
    elif img.ndim == 3:
        z = grids[0].ravel() - cen[0] * spacing[0]
        y = grids[1].ravel() - cen[1] * spacing[1]
        x = grids[2].ravel() - cen[2] * spacing[2]

        cxx = cov(x, x, img.ravel())
        cyy = cov(y, y, img.ravel())
        czz = cov(z, z, img.ravel())
        cxy = cov(x, y, img.ravel())
        cxz = cov(x, z, img.ravel())
        cyz = cov(y, z, img.ravel())

        return np.array([[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]])
    else:
        NotImplementedError()


def fwhm(sigma):
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def plot_fit_1d(psf1d, coords, params, prefix, ylim=[0, 9200]):
    fine_coords = np.linspace(coords[0], coords[-1], 500)
    plt.plot(coords, psf1d, "-", label="measurment", color="k")
    plt.bar(coords, psf1d, width=15, color="k")
    plt.plot(
        coords,
        [
            params[1],
        ]
        * len(coords),
        "--",
        label=f"{prefix} background",
    )
    plt.plot(
        coords,
        [
            params[1] + params[0],
        ]
        * len(coords),
        "--",
        label=f"{prefix} amplitude",
    )
    plt.plot(
        [
            params[2],
        ]
        * 2,
        [params[1], params[1] + params[0]],
        "--",
        label=f"{prefix} location",
    )
    plt.plot(
        fine_coords, gauss_1d(*params)(fine_coords), "--", label=f"{prefix} Gaussian"
    )
    plt.ylim(ylim)
    plt.legend()


def gauss_1d(amp, bg, mu, sigma):
    return lambda x: amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + bg


def eval_fun(x, amp, bg, mu, sigma):
    return gauss_1d(amp=amp, bg=bg, mu=mu, sigma=sigma)(x)


def get_objective_1d(coords, measurement):
    def objective(params):

        fit = gauss_1d(*params)(coords)

        kld = kl_div(fit, measurement)
        kld[np.isinf(kld)] = 0

        return np.sum(kld)

    return objective


def compute_KL_div_fit(coords_x, psf_x, params, spacing):
    res = minimize(
        get_objective_1d(coords_x, psf_x - params[1]),
        x0=params,
        method="L-BFGS-B",
        options={"disp": True},
        bounds=[
            (params[0] / 2, params[0] * 2),
            (0, params[0] / 2),
            (params[2] - params[3] / 2, params[2] + params[3] / 2),
            (spacing[1] / 4, 2 * params[3]),
        ],
    )

    res.x[1] += params[1]
    return res


def mse(params, coords, target):
    return np.mean((target - gauss_1d(*params)(coords)) ** 2)


def mae(params, coords, target):
    return np.mean(np.abs(target - gauss_1d(*params)(coords)))


def kl_div_error(params, coords, target, get_objective):
    return get_objective(coords, target)(params)


def gauss_2d(amp, bg, mu_x, mu_y, cxx, cxy, cyy):
    def fun(coords):
        cov_inv = np.linalg.inv(np.array([[cxx, cxy], [cxy, cyy]]))
        exponent = -0.5 * (
            cov_inv[0, 0] * (coords[:, 1] - mu_x) ** 2
            + 2 * cov_inv[0, 1] * (coords[:, 1] - mu_x) * (coords[:, 0] - mu_y)
            + cov_inv[1, 1] * (coords[:, 0] - mu_y) ** 2
        )

        return amp * np.exp(exponent) + bg

    return fun


def show_2d_fit(psf, fit):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(psf)
    plt.subplot(1, 2, 2)
    plt.imshow(fit)
    plt.show()


def eval_fun_2d(x, amp, bg, mu_x, mu_y, cxx, cxy, cyy):
    return gauss_2d(amp=amp, bg=bg, mu_x=mu_x, mu_y=mu_y, cxx=cxx, cxy=cxy, cyy=cyy)(x)


def get_objective_2d(coords, measurement):
    def objective(params):

        fit = gauss_2d(*params)(coords)

        kld = kl_div(fit, measurement)
        kld[np.isinf(kld)] = 0

        return np.sum(kld)

    return objective


def compute_KL_div_fit_2d(coords_yx, psf_yx, params, spacing):
    res = minimize(
        get_objective_2d(coords_yx, psf_yx - params[1]),
        x0=params,
        method="L-BFGS-B",
        options={"disp": True},
        bounds=[
            (params[0] * 0.5, params[0] * 1.5),
            (0, params[0] / 2),
            (params[2] - spacing[2], params[2] + spacing[2]),
            (params[3] - spacing[1], params[3] + spacing[1]),
            (params[4] * 0.5, params[4] * 1.5),
            (None, None),
            (params[6] * 0.5, params[6] * 1.5),
        ],
    )

    res.x[1] += params[1]
    return res


def gauss_3d(amp, bg, mu_x, mu_y, mu_z, cxx, cxy, cxz, cyy, cyz, czz):
    def fun(coords):
        cov_inv = np.linalg.inv(
            np.array([[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]])
        )
        exponent = -0.5 * (
            cov_inv[0, 0] * (coords[:, 2] - mu_x) ** 2
            + 2 * cov_inv[0, 1] * (coords[:, 2] - mu_x) * (coords[:, 1] - mu_y)
            + 2 * cov_inv[0, 2] * (coords[:, 2] - mu_x) * (coords[:, 0] - mu_z)
            + cov_inv[1, 1] * (coords[:, 1] - mu_y) ** 2
            + 2 * cov_inv[1, 2] * (coords[:, 1] - mu_y) * (coords[:, 0] - mu_z)
            + cov_inv[2, 2] * (coords[:, 0] - mu_z) ** 2
        )

        return amp * np.exp(exponent) + bg

    return fun


def eval_fun_3d(x, amp, bg, mu_x, mu_y, mu_z, cxx, cxy, cxz, cyy, cyz, czz):
    return gauss_3d(
        amp=amp,
        bg=bg,
        mu_x=mu_x,
        mu_y=mu_y,
        mu_z=mu_z,
        cxx=cxx,
        cxy=cxy,
        cxz=cxz,
        cyy=cyy,
        cyz=cyz,
        czz=czz,
    )(x)


def get_objective_3d(coords, measurement):
    def objective(params):

        fit = gauss_3d(*params)(coords)

        kld = kl_div(fit, measurement)
        kld[np.isinf(kld)] = 0

        return np.sum(kld)

    return objective


def compute_KL_div_fit_3d(coords_zyx, psf, params, spacing):
    res = minimize(
        get_objective_3d(coords_zyx, psf - params[1]),
        x0=params,
        method="L-BFGS-B",
        options={"disp": True},
        bounds=[
            (params[0] * 0.5, params[0] * 1.5),
            (0, params[0] / 2),
            (params[2] - spacing[2], params[2] + spacing[2]),
            (params[3] - spacing[1], params[3] + spacing[1]),
            (params[4] - spacing[0], params[4] + spacing[0]),
            (params[5] * 0.5, params[5] * 1.5),
            (None, None),
            (None, None),
            (params[8] * 0.5, params[8] * 1.5),
            (None, None),
            (params[10] * 0.5, params[10] * 1.5),
        ],
    )

    res.x[1] += params[1]
    return res


def get_ellipsoid(cov, spacing):
    bias = 0
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    x, y, z = (cov @ np.stack((x, y, z), 0).reshape(3, -1) + bias).reshape(3, *x.shape)
    return x / spacing[2], y / spacing[1], z / spacing[0]
