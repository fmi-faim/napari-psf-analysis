from copy import copy
from datetime import datetime
from os.path import basename

import numpy as np
import pandas as pd
from psf_fit_utils import (
    compute_KL_div_fit,
    compute_KL_div_fit_2d,
    compute_KL_div_fit_3d,
    eval_fun,
    eval_fun_2d,
    eval_fun_3d,
    fwhm,
    gauss_2d,
    gauss_3d,
    get_cov_matrix,
    get_raw_crop,
    plot_1d_fits,
    plot_3d_fits,
)
from scipy.optimize import curve_fit
from skimage.measure import centroid, find_contours
from tifffile import imread


def main():

    files = [
        ["W1_Levi_40x_1_Conf488.stk", [200, 163, 163]],
        ["20221114_PSF_Stellaris5_60X_1.tif", [200, 36, 36]],
        ["W1_Levi_63x_1_Conf488.stk", [200, 103, 103]],
    ]

    results = {
        "file_name": [],
        "spacing": [],
        "est_1d_fwhm_x": [],
        "curvefit_1d_fwhm_x": [],
        "curvefit_2d_fwhm_x": [],
        "curvefit_3d_fwhm_x": [],
        "kldiv_1d_fwhm_x": [],
        "kldiv_2d_fwhm_x": [],
        "kldiv_3d_fwhm_x": [],
        "est_1d_fwhm_y": [],
        "curvefit_1d_fwhm_y": [],
        "curvefit_2d_fwhm_y": [],
        "curvefit_3d_fwhm_y": [],
        "kldiv_1d_fwhm_y": [],
        "kldiv_2d_fwhm_y": [],
        "kldiv_3d_fwhm_y": [],
        "est_1d_fwhm_z": [],
        "curvefit_1d_fwhm_z": [],
        "curvefit_3d_fwhm_z": [],
        "kldiv_1d_fwhm_z": [],
        "kldiv_3d_fwhm_z": [],
        "curvefit_PA1": [],
        "curvefit_PA2": [],
        "curvefit_PA3": [],
        "kldiv_PA1": [],
        "kldiv_PA2": [],
        "kldiv_PA3": [],
    }

    for file_name, sp in files:
        results["file_name"].append(file_name)
        results["spacing"].append(sp)
        img = imread(file_name)
        spacing = np.array(sp)

        psf = get_raw_crop(img, 4000, spacing)
        y_lim = [0, psf.max() * 1.1]

        # 1D X
        psf_x = psf[psf.shape[0] // 2, psf.shape[1] // 2]
        coords_x = np.arange(len(psf_x)) * spacing[2]

        bg = np.median(psf)
        amp = psf_x.max() - bg
        sigma = np.sqrt(
            get_cov_matrix(np.clip(psf_x - bg, 0, psf_x.max()), [spacing[2]])
        )
        mu = centroid(psf_x) * spacing[2]

        params = [amp, bg, mu[0], sigma]
        popt, pcov = curve_fit(
            eval_fun,
            coords_x,
            psf_x,
            p0=params,
        )

        res = compute_KL_div_fit(coords_x, psf_x, params, spacing)
        mu_x = res.x[2]
        kld_fwhm_x = fwhm(res.x[3])

        results["est_1d_fwhm_x"].append(fwhm(sigma))
        results["curvefit_1d_fwhm_x"].append(fwhm(popt[3]))
        results["kldiv_1d_fwhm_x"].append(kld_fwhm_x)
        fine_coords_x = np.linspace(coords_x[0], coords_x[-1], 500)
        plot_1d_fits(
            basename(file_name) + "-1D-X",
            psf_x,
            coords_x,
            fine_coords_x,
            amp,
            bg,
            mu,
            sigma,
            popt,
            res,
            fwhm(sigma),
            fwhm(popt[3]),
            kld_fwhm_x,
            y_lim,
        )

        # 1D Y
        psf_y = psf[psf.shape[0] // 2, :, psf.shape[2] // 2]
        coords_y = np.arange(len(psf_y)) * spacing[1]

        bg = np.median(psf)
        amp = psf_y.max() - bg
        sigma = np.sqrt(
            get_cov_matrix(np.clip(psf_y - bg, 0, psf_y.max()), [spacing[1]])
        )
        mu = centroid(psf_y) * spacing[1]

        params = [amp, bg, mu[0], sigma]
        popt, pcov = curve_fit(
            eval_fun,
            coords_y,
            psf_y,
            p0=params,
        )

        res = compute_KL_div_fit(coords_y, psf_y, params, spacing)
        mu_y = res.x[2]
        kld_fwhm_y = fwhm(res.x[3])

        results["est_1d_fwhm_y"].append(fwhm(sigma))
        results["curvefit_1d_fwhm_y"].append(fwhm(popt[3]))
        results["kldiv_1d_fwhm_y"].append(kld_fwhm_y)
        fine_coords_y = np.linspace(coords_y[0], coords_y[-1], 500)
        plot_1d_fits(
            basename(file_name) + "-1D-Y",
            psf_y,
            coords_y,
            fine_coords_y,
            amp,
            bg,
            mu,
            sigma,
            popt,
            res,
            fwhm(sigma),
            fwhm(popt[3]),
            kld_fwhm_y,
            y_lim,
        )

        # 1D Z
        psf_z = psf[:, psf.shape[1] // 2, psf.shape[2] // 2]
        coords_z = np.arange(len(psf_z)) * spacing[0]

        bg = np.median(psf)
        amp = psf_z.max() - bg
        sigma = np.sqrt(
            get_cov_matrix(np.clip(psf_z - bg, 0, psf_z.max()), [spacing[0]])
        )
        mu = centroid(psf_z) * spacing[0]

        params = [amp, bg, mu[0], sigma]
        popt, pcov = curve_fit(
            eval_fun,
            coords_z,
            psf_z,
            p0=params,
        )

        res = compute_KL_div_fit(coords_z, psf_z, params, spacing)
        mu_z = res.x[2]
        kld_fwhm_z = fwhm(res.x[3])

        results["est_1d_fwhm_z"].append(fwhm(sigma))
        results["curvefit_1d_fwhm_z"].append(fwhm(popt[3]))
        results["kldiv_1d_fwhm_z"].append(kld_fwhm_z)
        fine_coords_z = np.linspace(coords_z[0], coords_z[-1], 500)
        plot_1d_fits(
            basename(file_name) + "-1D-Z",
            psf_z,
            coords_z,
            fine_coords_z,
            amp,
            bg,
            mu,
            sigma,
            popt,
            res,
            fwhm(sigma),
            fwhm(popt[3]),
            kld_fwhm_z,
            y_lim,
        )

        # 2D XY
        psf_yx = psf[psf.shape[0] // 2]
        yy = np.arange(psf_yx.shape[0]) * spacing[1]
        xx = np.arange(psf_yx.shape[1]) * spacing[2]
        y, x = np.meshgrid(yy, xx, indexing="ij")
        coords_yx = np.stack([y.ravel(), x.ravel()], -1)

        yy_fine = np.linspace(0, psf_yx.shape[0], 500) * spacing[1]
        xx_fine = np.linspace(0, psf_yx.shape[1], 500) * spacing[2]
        y_fine, x_fine = np.meshgrid(yy_fine, xx_fine, indexing="ij")
        fine_coords_yx = np.stack([y_fine.ravel(), x_fine.ravel()], -1)

        fine_coords_y = np.stack(
            [
                yy_fine,
                [
                    xx_fine[xx_fine.shape[0] // 2],
                ]
                * len(yy_fine),
            ],
            axis=1,
        )
        fine_coords_x = np.stack(
            [
                [
                    yy_fine[yy_fine.shape[0] // 2],
                ]
                * len(xx_fine),
                xx_fine,
            ],
            axis=1,
        )

        bg = np.median(psf)
        amp = psf_yx.max() - bg

        params = [
            amp,
            bg,
            mu_x,
            mu_y,
            (kld_fwhm_x / (2 * np.sqrt(2 * np.log(2)))) ** 2,
            0,
            (kld_fwhm_y / (2 * np.sqrt(2 * np.log(2)))) ** 2,
        ]

        popt, pcov = curve_fit(
            eval_fun_2d,
            coords_yx,
            psf_yx.ravel(),
            p0=params,
        )
        cv_2d_params = copy(popt)

        res = compute_KL_div_fit_2d(coords_yx, psf_yx.ravel(), params, spacing)
        kl_2d_params = copy(res.x)

        results["curvefit_2d_fwhm_x"].append(fwhm(np.sqrt(popt[4])))
        results["kldiv_2d_fwhm_x"].append(fwhm(np.sqrt(res.x[4])))
        results["curvefit_2d_fwhm_y"].append(np.sqrt(fwhm(popt[6])))
        results["kldiv_2d_fwhm_y"].append(np.sqrt(fwhm(res.x[6])))

        # 3D
        zz = np.arange(psf.shape[0]) * spacing[0]
        yy = np.arange(psf.shape[1]) * spacing[1]
        xx = np.arange(psf.shape[2]) * spacing[2]
        z, y, x = np.meshgrid(zz, yy, xx, indexing="ij")
        coords_zyx = np.stack([z.ravel(), y.ravel(), x.ravel()], -1)

        zz_fine = np.linspace(0, psf.shape[0], 500) * spacing[0]
        yy_fine = np.linspace(0, psf.shape[1], 500) * spacing[1]
        xx_fine = np.linspace(0, psf.shape[2], 500) * spacing[2]
        z_fine, y_fine, x_fine = np.meshgrid(zz_fine, yy_fine, xx_fine, indexing="ij")

        bg = np.median(psf)
        amp = psf.max() - bg

        params = [
            amp,
            bg,
            mu_x,
            mu_y,
            mu_z,
            (kld_fwhm_x / (2 * np.sqrt(2 * np.log(2)))) ** 2,
            0,
            0,
            (kld_fwhm_y / (2 * np.sqrt(2 * np.log(2)))) ** 2,
            0,
            (kld_fwhm_z / (2 * np.sqrt(2 * np.log(2)))) ** 2,
        ]

        popt, pcov = curve_fit(
            eval_fun_3d,
            coords_zyx,
            psf.ravel(),
            p0=params,
        )

        res = compute_KL_div_fit_3d(coords_zyx, psf.ravel(), params, spacing)

        cv_3d_fwhm_x = fwhm(np.sqrt(popt[5]))
        cv_3d_fwhm_y = fwhm(np.sqrt(popt[8]))
        cv_3d_fwhm_z = fwhm(np.sqrt(popt[10]))

        kld_3d_fwhm_x = fwhm(np.sqrt(res.x[5]))
        kld_3d_fwhm_y = fwhm(np.sqrt(res.x[8]))
        kld_3d_fwhm_z = fwhm(np.sqrt(res.x[10]))

        results["curvefit_3d_fwhm_x"].append(cv_3d_fwhm_x)
        results["curvefit_3d_fwhm_y"].append(cv_3d_fwhm_y)
        results["curvefit_3d_fwhm_z"].append(cv_3d_fwhm_z)
        results["kldiv_3d_fwhm_x"].append(kld_3d_fwhm_x)
        results["kldiv_3d_fwhm_y"].append(kld_3d_fwhm_y)
        results["kldiv_3d_fwhm_z"].append(kld_3d_fwhm_z)

        cv_3d_cov = np.array(
            [
                [popt[5], popt[6], popt[7]],
                [popt[6], popt[8], popt[9]],
                [popt[7], popt[9], popt[10]],
            ]
        )

        kl_3d_cov = np.array(
            [
                [res.x[5], res.x[6], res.x[7]],
                [res.x[6], res.x[8], res.x[9]],
                [res.x[7], res.x[9], res.x[10]],
            ]
        )

        pc = np.sort(np.sqrt(np.linalg.eigvals(cv_3d_cov)))[::-1]
        cv_3d_pa1 = fwhm(pc[0])
        cv_3d_pa2 = fwhm(pc[1])
        cv_3d_pa3 = fwhm(pc[2])

        pc = np.sort(np.sqrt(np.linalg.eigvals(kl_3d_cov)))[::-1]
        kl_3d_pa1 = fwhm(pc[0])
        kl_3d_pa2 = fwhm(pc[1])
        kl_3d_pa3 = fwhm(pc[2])

        results["curvefit_PA1"].append(cv_3d_pa1)
        results["curvefit_PA2"].append(cv_3d_pa2)
        results["curvefit_PA3"].append(cv_3d_pa3)
        results["kldiv_PA1"].append(kl_3d_pa1)
        results["kldiv_PA2"].append(kl_3d_pa2)
        results["kldiv_PA3"].append(kl_3d_pa3)

        # Plot
        fc_z_plane = np.stack(
            [
                (np.ones((1, 500, 500)) * psf.shape[0] // 2 * spacing[0]).ravel(),
                y_fine[:1].ravel(),
                x_fine[:1].ravel(),
            ],
            -1,
        )
        fc_y_plane = np.stack(
            [
                z_fine[:, :1].ravel(),
                (np.ones((500, 1, 500)) * psf.shape[1] // 2 * spacing[1]).ravel(),
                x_fine[:, :1].ravel(),
            ],
            -1,
        )
        fc_x_plane = np.stack(
            [
                z_fine[:, :, :1].ravel(),
                y_fine[:, :, :1].ravel(),
                (np.ones((500, 500, 1)) * psf.shape[1] // 2 * spacing[2]).ravel(),
            ],
            -1,
        )

        base_z_plane = gauss_3d(*params)(fc_z_plane).reshape(500, 500)
        cv_z_plane = gauss_3d(*popt)(fc_z_plane).reshape(500, 500)
        kl_z_plane = gauss_3d(*res.x)(fc_z_plane).reshape(500, 500)

        base_y_plane = gauss_3d(*params)(fc_y_plane).reshape(500, 500)
        cv_y_plane = gauss_3d(*popt)(fc_y_plane).reshape(500, 500)
        kl_y_plane = gauss_3d(*res.x)(fc_y_plane).reshape(500, 500)

        base_x_plane = gauss_3d(*params)(fc_x_plane).reshape(500, 500)
        cv_x_plane = gauss_3d(*popt)(fc_x_plane).reshape(500, 500)
        kl_x_plane = gauss_3d(*res.x)(fc_x_plane).reshape(500, 500)

        cv2d_z_plane = gauss_2d(*cv_2d_params)(fine_coords_yx).reshape(500, 500)
        kl2d_z_plane = gauss_2d(*kl_2d_params)(fine_coords_yx).reshape(500, 500)

        base_z_contours = find_contours(base_z_plane, level=params[0] / 2 + params[1])[
            0
        ]
        base_z_contours = base_z_contours / 500 * psf.shape[1]

        cv2d_z_contours = find_contours(
            cv2d_z_plane, level=cv_2d_params[0] / 2 + cv_2d_params[1]
        )[0]
        cv2d_z_contours = cv2d_z_contours / 500 * psf.shape[1]

        kl2d_z_contours = find_contours(
            kl2d_z_plane, level=kl_2d_params[0] / 2 + kl_2d_params[1]
        )[0]
        kl2d_z_contours = kl2d_z_contours / 500 * psf.shape[1]

        base_y_contours = find_contours(base_y_plane, level=params[0] / 2 + params[1])[
            0
        ]
        base_y_contours = base_y_contours / 500 * psf.shape[1]

        base_x_contours = find_contours(base_x_plane, level=params[0] / 2 + params[1])[
            0
        ]
        base_x_contours = base_x_contours / 500 * psf.shape[1]

        cv_z_contours = find_contours(cv_z_plane, level=popt[0] / 2 + popt[1])[0]
        cv_z_contours = cv_z_contours / 500 * psf.shape[1]

        cv_y_contours = find_contours(cv_y_plane, level=popt[0] / 2 + popt[1])[0]
        cv_y_contours = cv_y_contours / 500 * psf.shape[1]

        cv_x_contours = find_contours(cv_x_plane, level=popt[0] / 2 + popt[1])[0]
        cv_x_contours = cv_x_contours / 500 * psf.shape[1]

        kl_z_contours = find_contours(kl_z_plane, level=res.x[0] / 2 + res.x[1])[0]
        kl_z_contours = kl_z_contours / 500 * psf.shape[1]

        kl_y_contours = find_contours(kl_y_plane, level=res.x[0] / 2 + res.x[1])[0]
        kl_y_contours = kl_y_contours / 500 * psf.shape[1]

        kl_x_contours = find_contours(kl_x_plane, level=res.x[0] / 2 + res.x[1])[0]
        kl_x_contours = kl_x_contours / 500 * psf.shape[1]

        plot_3d_fits(
            file_name,
            psf,
            spacing,
            params,
            popt,
            res,
            base_z_contours,
            cv_z_contours,
            kl_z_contours,
            base_y_contours,
            cv_y_contours,
            kl_y_contours,
            base_x_contours,
            cv_x_contours,
            kl_x_contours,
        )

    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{datetime.today().strftime('%Y-%m-%d_%H:%M:%S')}-results.csv")


if __name__ == "__main__":
    main()
