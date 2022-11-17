import datetime

import numpy as np
import pandas as pd
import pkg_resources
from napari.utils.notifications import show_info
from scipy.optimize import curve_fit
from skimage.filters import gaussian
from skimage.measure import centroid


class PSFAnalysis:
    """
    Extract and measure point spread functions (PSFs) of a bead image.
    Only the beads indicated with a point are measured.
    """

    def __init__(
        self,
        date: datetime.datetime,
        microscope: str,
        magnification: int,
        NA: float,
        spacing: np.array,
        patch_size: np.array,
    ):
        """
        Parameters:
            date: datetime.datetime
                Date of acquisition.
            microscope: str
                Name of the acquisition microscope.
            magnification: float
                Used magnification to image the bead.
            NA: float
                Numerical apperture of the objective.
            spacing: np.array([float, float, float])
                Voxel size (Z, Y, X)
            patch_size: np.array
                Sub-volume to extract around the bead and analyze.
        """
        self.date = date
        self.microscope = microscope
        self.magnification = magnification
        self.NA = NA
        # Necessary since we report XYpixelspacing as a single value.
        assert spacing[1] == spacing[2], "Pixel spacing needs to be square in XY."
        self.spacing = spacing
        self.patch_size = np.array(patch_size)

    @staticmethod
    def _create_slice(mean, margin):
        return slice(int(np.round(mean - margin)), int(np.round(mean + margin)))

    def _localize_beads(self, img, points):
        beads = []
        offsets = []
        margins = self.patch_size / self.spacing / 2
        for p in points:
            if np.all(p > margins) and np.all(p < (np.array(img.shape) - margins)):
                z_search_slice = self._create_slice(p[0], margins[0])
                y_search_slice = self._create_slice(p[1], margins[1])
                x_search_slice = self._create_slice(p[2], margins[2])
                subvolume = img[z_search_slice, y_search_slice, x_search_slice]

                closest_roi = np.unravel_index(
                    np.argmax(
                        gaussian(subvolume, 2, mode="constant", preserve_range=True)
                    ),
                    subvolume.shape,
                )

                z_slice = self._create_slice(
                    closest_roi[0] + z_search_slice.start, margins[0]
                )
                y_slice = self._create_slice(
                    closest_roi[1] + y_search_slice.start, margins[1]
                )
                x_slice = self._create_slice(
                    closest_roi[2] + x_search_slice.start, margins[2]
                )

                out_z = (z_slice.start < 0) or (z_slice.stop >= img.shape[0])
                out_y = (y_slice.start < 0) or (y_slice.stop >= img.shape[1])
                out_x = (x_slice.start < 0) or (x_slice.stop >= img.shape[2])

                if out_z or out_y or out_x:
                    show_info(
                        "Discarded point ({}, {}, {}). Too close to image border.".format(
                            np.round(p[2]),
                            np.round(p[1]),
                            np.round(p[0]),
                        )
                    )
                else:
                    bead = img[z_slice, y_slice, x_slice]
                    offsets.append(tuple([z_slice.start, y_slice.start, x_slice.start]))
                    beads.append(bead)
            else:
                show_info(
                    "Discarded point ({}, {}, {}). Too close to image border.".format(
                        np.round(p[2]),
                        np.round(p[1]),
                        np.round(p[0]),
                    )
                )

        return beads, offsets

    @staticmethod
    def ellipsoid3D(data, A, B, mu_x, mu_y, mu_z, cxx, cxy, cxz, cyy, cyz, czz):
        inv = np.linalg.inv(
            np.array([[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]])
            + np.identity(3) * 1e-6
        )
        return (
            A
            * np.exp(
                -0.5
                * (
                    inv[0, 0] * (data[:, 2] - mu_x) ** 2
                    + 2 * inv[0, 1] * (data[:, 2] - mu_x) * (data[:, 1] - mu_y)
                    + 2 * inv[0, 2] * (data[:, 2] - mu_x) * (data[:, 0] - mu_z)
                    + inv[1, 1] * (data[:, 1] - mu_y) ** 2
                    + 2 * inv[1, 2] * (data[:, 1] - mu_y) * (data[:, 0] - mu_z)
                    + inv[2, 2] * (data[:, 0] - mu_z) ** 2
                )
            )
            + B
        )

    @staticmethod
    def ellipsoid2D(data, A, B, mu_x, mu_y, cxx, cxy, cyy):
        inv = np.linalg.inv(np.array([[cxx, cxy], [cxy, cyy]]) + np.identity(2) * 1e-8)

        return (
            A
            * np.exp(
                -0.5
                * (
                    inv[0, 0] * (data[:, 1] - mu_x) ** 2
                    + 2 * inv[0, 1] * (data[:, 1] - mu_x) * (data[:, 0] - mu_y)
                    + inv[1, 1] * (data[:, 0] - mu_y) ** 2
                )
            )
            + B
        )

    @staticmethod
    def get_estimates_2D(data, spacing):
        max_ = data.max()
        mean_ = data.mean()
        cy, cx = centroid(data)
        cy *= spacing[1]
        cx *= spacing[2]
        cov = PSFAnalysis.get_cov_matrix_2D(data, spacing)
        return [
            max_ - mean_,
            mean_,
            cx,
            cy,
            cov[0, 0],
            cov[0, 1],
            cov[1, 1],
        ]

    @staticmethod
    def get_estimates(data, spacing):
        max_ = data.max()
        mean_ = data.mean()
        cz, cy, cx = centroid(data)
        cz *= spacing[0]
        cy *= spacing[1]
        cx *= spacing[2]
        cov = PSFAnalysis.get_cov_matrix(data, spacing)
        return [
            max_ - mean_,
            mean_,
            cx,
            cy,
            cz,
            cov[0, 0],
            cov[0, 1],
            cov[0, 2],
            cov[1, 1],
            cov[1, 2],
            cov[2, 2],
        ]

    @staticmethod
    def get_cov_matrix_2D(img, spacing):
        def cov(x, y, i):
            return np.sum(x * y * i) / np.sum(i)

        y, x = np.meshgrid(
            np.arange(img.shape[0]) * spacing[0],
            np.arange(img.shape[1]) * spacing[1],
            indexing="ij",
        )
        cen = centroid(img)
        y = y.ravel() - cen[0] * spacing[0]
        x = x.ravel() - cen[1] * spacing[1]

        cxx = cov(x, x, img.ravel())
        cyy = cov(y, y, img.ravel())
        cxy = cov(x, y, img.ravel())

        C = np.array([[cxx, cxy], [cxy, cyy]])

        return C

    @staticmethod
    def get_cov_matrix(img, spacing):
        def cov(x, y, i):
            return np.sum(x * y * i) / np.sum(i)

        z, y, x = np.meshgrid(
            np.arange(img.shape[0]) * spacing[0],
            np.arange(img.shape[1]) * spacing[1],
            np.arange(img.shape[2]) * spacing[2],
            indexing="ij",
        )
        cen = centroid(img)
        z = z.ravel() - cen[0] * spacing[0]
        y = y.ravel() - cen[1] * spacing[1]
        x = x.ravel() - cen[2] * spacing[2]

        cxx = cov(x, x, img.ravel())
        cyy = cov(y, y, img.ravel())
        czz = cov(z, z, img.ravel())
        cxy = cov(x, y, img.ravel())
        cxz = cov(x, z, img.ravel())
        cyz = cov(y, z, img.ravel())

        C = np.array([[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]])

        return C

    @staticmethod
    def _fit_gaussian_3d(data, spacing):

        crop_xy = max(0, int((data.shape[1] - (3000 / spacing[1])) // 2))
        if crop_xy > 0:
            data_cropped = data[:, crop_xy:-crop_xy, crop_xy:-crop_xy]
        else:
            data_cropped = data

        zz = np.arange(data_cropped.shape[0]) * spacing[0]
        yy = np.arange(data_cropped.shape[1]) * spacing[1]
        xx = np.arange(data_cropped.shape[2]) * spacing[2]
        z, y, x = np.meshgrid(zz, yy, xx, indexing="ij")
        coords = np.stack([z.ravel(), y.ravel(), x.ravel()], -1)

        popt, pcov = curve_fit(
            PSFAnalysis.ellipsoid3D,
            coords,
            data_cropped.ravel(),
            p0=PSFAnalysis.get_estimates(data_cropped, spacing),
            bounds=(
                [
                    np.mean(data_cropped),
                    0,
                    0,
                    0,
                    0,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                ],
                [
                    np.inf,
                    data.max(),
                    data.shape[2] * spacing[2],
                    data.shape[1] * spacing[1],
                    data.shape[0] * spacing[0],
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                ],
            ),
        )

        popt[2] += crop_xy * spacing[2]
        popt[3] += crop_xy * spacing[1]

        return popt, pcov

    @staticmethod
    def _fit_gaussian_2d(data, spacing):

        crop_xy = max(0, int((data.shape[0] - (3000 / spacing[1])) // 2))

        if crop_xy > 0:
            data_cropped = data[crop_xy:-crop_xy, crop_xy:-crop_xy]
        else:
            data_cropped = data

        yy = np.arange(data_cropped.shape[0]) * spacing[1]
        xx = np.arange(data_cropped.shape[1]) * spacing[2]
        y, x = np.meshgrid(yy, xx, indexing="ij")
        coords = np.stack([y.ravel(), x.ravel()], -1)
        popt, pcov = curve_fit(
            PSFAnalysis.ellipsoid2D,
            coords,
            data_cropped.ravel(),
            p0=PSFAnalysis.get_estimates_2D(data_cropped, spacing),
            bounds=(
                [np.mean(data), 0, 0, 0, -np.inf, -np.inf, -np.inf],
                [
                    np.inf,
                    data.max(),
                    data.shape[1] * spacing[2],
                    data.shape[0] * spacing[1],
                    np.inf,
                    np.inf,
                    np.inf,
                ],
            ),
        )

        popt[2] += crop_xy * spacing[2]
        popt[3] += crop_xy * spacing[1]

        return popt, pcov

    @staticmethod
    def _fwhm(sigma):
        return 2 * np.sqrt(2 * np.log(2)) * sigma

    @staticmethod
    def _get_signal(bead, mu_z, mu_y, mu_x):
        z_slice = bead[int(np.round(mu_z))]
        y = int(np.round(mu_y))
        x = int(np.round(mu_x))
        return (
            z_slice[y - 1, x - 1]
            + z_slice[y - 1, x]
            + z_slice[y - 1, x + 1]
            + z_slice[y, x - 1]
            + z_slice[y, x]
            + z_slice[y, x + 1]
            + z_slice[y + 1, x - 1]
            + z_slice[y + 1, x]
            + z_slice[y + 1, x + 1]
        ) / 9.0

    def analyze(self, img_name: str, img: np.array, points: list):
        """
        Analyze beads, indicated by points, in a given image.

        Parameters:
            img_name: str
                Name of the image used in the results table.
            img: np.array
                Image data.
            points: list
                Point coordinates.
        """
        beads, offsets = self._localize_beads(img, points)
        popts, pcovs = [], []
        popts_2d, pcovs_2d = [], []
        fitted_beads, fitted_offsets = [], []
        for bead, offset in zip(beads, offsets):
            try:
                popt, pcov = self._fit_gaussian_3d(bead, self.spacing)
                focus_plane = bead[int(np.round(popt[4] // self.spacing[0]))]
                popt_2d, pcov_2d = self._fit_gaussian_2d(focus_plane, self.spacing)

                popts.append(popt)
                pcovs.append(pcov)
                popts_2d.append(popt_2d)
                pcovs_2d.append(pcov_2d)
                fitted_beads.append(bead)
                fitted_offsets.append(offset)
            except RuntimeError as e:
                if "Optimal parameters not found:" in str(e):
                    z_pos = offset[0] + bead.shape[0] // 2
                    y_pos = offset[1] + bead.shape[1] // 2
                    x_pos = offset[2] + bead.shape[2] // 2
                    show_info(
                        f"No fit found at position ({z_pos}, {y_pos},"
                        f" {x_pos}). Discarding point."
                    )
                    continue

        results = self._create_results_table(
            fitted_beads,
            popts,
            pcovs,
            popts_2d,
            pcovs_2d,
            img_name,
            fitted_offsets,
        )
        return (fitted_beads, popts, pcovs, popts_2d, pcovs_2d, results)

    def _create_results_table(
        self,
        beads,
        popts,
        pcovs,
        popts_2d,
        pcovs_2d,
        img_name,
        offsets,
    ):
        c_image_name = []
        c_date = []
        c_microscope = []
        c_mag = []
        c_na = []
        c_amp = []
        c_amp_2d = []
        c_background = []
        c_background_2d = []
        c_x = []
        c_y = []
        c_z = []
        c_x_2d = []
        c_y_2d = []
        c_fwhm_x = []
        c_fwhm_y = []
        c_fwhm_z = []
        c_fwhm_2d_x = []
        c_fwhm_2d_y = []
        c_pa1 = []
        c_pa2 = []
        c_pa3 = []
        c_pa1_2d = []
        c_pa2_2d = []
        c_s2bg = []
        c_s2bg_2d = []
        c_xyspacing = []
        c_zspacing = []
        c_cxx = []
        c_cxy = []
        c_cxz = []
        c_cyy = []
        c_cyz = []
        c_czz = []
        c_cxx_2d = []
        c_cxy_2d = []
        c_cyy_2d = []
        c_sde_peak = []
        c_sde_background = []
        c_sde_mu_x = []
        c_sde_mu_y = []
        c_sde_mu_z = []
        c_sde_cxx = []
        c_sde_cxy = []
        c_sde_cxz = []
        c_sde_cyy = []
        c_sde_cyz = []
        c_sde_czz = []
        c_sde_peak_2d = []
        c_sde_background_2d = []
        c_sde_mu_x_2d = []
        c_sde_mu_y_2d = []
        c_sde_cxx_2d = []
        c_sde_cxy_2d = []
        c_sde_cyy_2d = []
        c_version = []
        for popt, pcov, popt_2d, pcov_2d, bead, offset in zip(
            popts, pcovs, popts_2d, pcovs_2d, beads, offsets
        ):
            perr = np.sqrt(np.diag(pcov))
            height = popt[0]
            background = popt[1]
            mu_x = popt[2]
            mu_y = popt[3]
            mu_z = popt[4]
            cxx = popt[5]
            cxy = popt[6]
            cxz = popt[7]
            cyy = popt[8]
            cyz = popt[9]
            czz = popt[10]
            cov = np.array([[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]])
            eigval, eigvec = np.linalg.eig(cov)
            pa3, pa2, pa1 = np.sort(np.sqrt(eigval))
            fwhm_x = self._fwhm(np.sqrt(cxx))
            fwhm_y = self._fwhm(np.sqrt(cyy))
            fwhm_z = self._fwhm(np.sqrt(czz))

            perr_2d = np.sqrt(np.diag(pcov))
            height_2d = popt_2d[0]
            background_2d = popt_2d[1]
            mu_x_2d = popt_2d[2]
            mu_y_2d = popt_2d[3]
            cxx_2d = popt_2d[4]
            cxy_2d = popt_2d[5]
            cyy_2d = popt_2d[6]
            cov_2d = np.array([[cxx_2d, cxy_2d], [cxy_2d, cyy_2d]])
            eigval_2d, eigvec_2d = np.linalg.eig(cov_2d)
            pa2_2d, pa1_2d = np.sort(np.sqrt(eigval_2d))
            fwhm_x_2d = self._fwhm(np.sqrt(cxx_2d))
            fwhm_y_2d = self._fwhm(np.sqrt(cyy_2d))

            c_image_name.append(img_name)
            c_date.append(self.date.strftime("%Y-%m-%d"))
            c_microscope.append(self.microscope)
            c_mag.append(self.magnification)
            c_na.append(self.NA)
            c_amp.append(height)
            c_amp_2d.append(height_2d)
            c_background.append(background)
            c_background_2d.append(background_2d)
            c_x.append(mu_x + offset[2])
            c_y.append(mu_y + offset[1])
            c_z.append(mu_z + offset[0])
            c_x_2d.append(mu_x_2d + offset[2])
            c_y_2d.append(mu_y_2d + offset[1])
            c_fwhm_x.append(fwhm_x)
            c_fwhm_y.append(fwhm_y)
            c_fwhm_z.append(fwhm_z)
            c_fwhm_2d_x.append(fwhm_x_2d)
            c_fwhm_2d_y.append(fwhm_y_2d)
            c_pa1.append(self._fwhm(pa1))
            c_pa2.append(self._fwhm(pa2))
            c_pa3.append(self._fwhm(pa3))
            c_pa1_2d.append(self._fwhm(pa1_2d))
            c_pa2_2d.append(self._fwhm(pa2_2d))
            c_s2bg.append(height / background)
            c_s2bg_2d.append(height_2d / background_2d)
            c_xyspacing.append(self.spacing[1])
            c_zspacing.append(self.spacing[0])
            c_cxx.append(cxx)
            c_cxy.append(cxy)
            c_cxz.append(cxz)
            c_cyy.append(cyy)
            c_cyz.append(cyz)
            c_czz.append(czz)
            c_cxx_2d.append(cxx_2d)
            c_cxy_2d.append(cxy_2d)
            c_cyy_2d.append(cyy_2d)

            c_sde_peak.append(perr[0])
            c_sde_background.append(perr[1])
            c_sde_mu_x.append(perr[2])
            c_sde_mu_y.append(perr[3])
            c_sde_mu_z.append(perr[4])
            c_sde_cxx.append(perr[5])
            c_sde_cxy.append(perr[6])
            c_sde_cxz.append(perr[7])
            c_sde_cyy.append(perr[8])
            c_sde_cyz.append(perr[9])
            c_sde_czz.append(perr[10])

            c_sde_peak_2d.append(perr_2d[0])
            c_sde_background_2d.append(perr_2d[1])
            c_sde_mu_x_2d.append(perr_2d[2])
            c_sde_mu_y_2d.append(perr_2d[3])
            c_sde_cxx_2d.append(perr_2d[4])
            c_sde_cxy_2d.append(perr_2d[5])
            c_sde_cyy_2d.append(perr_2d[6])

            c_version.append(
                pkg_resources.get_distribution("napari_psf_analysis").version
            )
        results = pd.DataFrame(
            {
                "ImageName": c_image_name,
                "Date": c_date,
                "Microscope": c_microscope,
                "Magnification": c_mag,
                "NA": c_na,
                "Amplitude": c_amp,
                "Amplitude_2D": c_amp_2d,
                "Background": c_background,
                "Background_2D": c_background_2d,
                "X": c_x,
                "Y": c_y,
                "Z": c_z,
                "X_2D": c_x_2d,
                "Y_2D": c_y_2d,
                "FWHM_X": c_fwhm_x,
                "FWHM_Y": c_fwhm_y,
                "FWHM_Z": c_fwhm_z,
                "FWHM_X_2D": c_fwhm_2d_x,
                "FWHM_Y_2D": c_fwhm_2d_y,
                "PrincipalAxis_1": c_pa1,
                "PrincipalAxis_2": c_pa2,
                "PrincipalAxis_3": c_pa3,
                "PrincipalAxis_1_2D": c_pa1_2d,
                "PrincipalAxis_2_2D": c_pa2_2d,
                "SignalToBG": c_s2bg,
                "SignalToBG_2D": c_s2bg_2d,
                "XYpixelsize": c_xyspacing,
                "Zspacing": c_zspacing,
                "cov_xx": c_cxx,
                "cov_xy": c_cxy,
                "cov_xz": c_cxz,
                "cov_yy": c_cyy,
                "cov_yz": c_cyz,
                "cov_zz": c_czz,
                "cov_xx_2D": c_cxx_2d,
                "cov_xy_2D": c_cxy_2d,
                "cov_yy_2D": c_cyy_2d,
                "sde_peak": c_sde_peak,
                "sde_background": c_sde_background,
                "sde_X": c_sde_mu_x,
                "sde_Y": c_sde_mu_y,
                "sde_Z": c_sde_mu_z,
                "sde_cov_xx": c_sde_cxx,
                "sde_cov_xy": c_sde_cxy,
                "sde_cov_xz": c_sde_cxz,
                "sde_cov_yy": c_sde_cyy,
                "sde_cov_yz": c_sde_cyz,
                "sde_cov_zz": c_sde_czz,
                "sde_peak_2D": c_sde_peak_2d,
                "sde_background_2D": c_sde_background_2d,
                "sde_X_2D": c_sde_mu_x_2d,
                "sde_Y_2D": c_sde_mu_y_2d,
                "sde_cov_xx_2D": c_sde_cxx_2d,
                "sde_cov_xy_2D": c_sde_cxy_2d,
                "sde_cov_yy_2D": c_sde_cyy_2d,
                "version": c_version,
            }
        )
        return results
