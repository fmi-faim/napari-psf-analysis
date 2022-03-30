import datetime

import numpy as np
import pandas as pd
from napari.utils.notifications import show_info
from scipy import optimize
from skimage.feature import peak_local_max
from skimage.filters import gaussian

from ..utils.gaussians import gaussian_1d, gaussian_3d


class PSFAnalysis:
    """
    Extract and measure point spread functions (PSFs) of a bead image.
    Only the beads indicated with a point are measured.
    """

    def __init__(
        self,
        date: datetime.datetime,
        microscope: str,
        magnification: float,
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
        smooth = gaussian(img, 2, mode="constant", preserve_range=True)
        coordinates = peak_local_max(
            smooth, min_distance=max(1, int(self.spacing[-1] // 4))
        )

        beads = []
        offsets = []
        margins = self.patch_size / self.spacing / 2
        for p in points:

            closest_roi = coordinates[
                np.argmin(np.linalg.norm(coordinates - p, axis=1))
            ]

            if np.all(closest_roi > margins) and np.all(
                closest_roi < (np.array(img.shape) - margins)
            ):
                z_slice = self._create_slice(closest_roi[0], margins[0])
                y_slice = self._create_slice(closest_roi[1], margins[1])
                x_slice = self._create_slice(closest_roi[2], margins[2])

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
    def _guess_init_params(data):
        data = data.astype(np.uint32)
        total = data.sum()
        Z, Y, X = np.indices(data.shape)
        z = (Z * data).sum() / total
        y = (Y * data).sum() / total
        x = (X * data).sum() / total

        z_slice = data[:, int(y), int(x)]
        y_slice = data[int(z), :, int(x)]
        x_slice = data[int(z), int(y), :]

        sigma_z = np.sqrt(
            np.abs((np.arange(z_slice.shape[0]) - z) ** 2 * z_slice).sum()
            / z_slice.sum()
        )
        sigma_y = np.sqrt(
            np.abs((np.arange(y_slice.shape[0]) - y) ** 2 * y_slice).sum()
            / y_slice.sum()
        )
        sigma_x = np.sqrt(
            np.abs((np.arange(x_slice.shape[0]) - x) ** 2 * x_slice).sum()
            / x_slice.sum()
        )

        offset = np.quantile(data, 0.5)
        height = data.max() - offset
        return height, z, y, x, sigma_z, sigma_y, sigma_x, offset

    @staticmethod
    def _get_loss_function(data):
        indices = np.indices(data.shape)

        def loss(p):
            return np.ravel(gaussian_3d(*p)(*indices) - data)

        return loss

    @staticmethod
    def _fit_gaussian_3d(data):
        loss_function = PSFAnalysis._get_loss_function(data)
        params, _ = optimize.leastsq(
            loss_function, PSFAnalysis._guess_init_params(data), full_output=False
        )
        return params

    @staticmethod
    def _r_squared(sample, height, mu, sigma, offset):
        gaussian = gaussian_1d(height, mu, sigma, offset)
        fit = gaussian(np.arange(sample.size))
        return 1 - np.sum((fit - sample) ** 2) / np.sum((fit - mu) ** 2)

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
        fitted_params = [self._fit_gaussian_3d(bead) for bead in beads]
        results = self._create_results_table(beads, fitted_params, img_name, offsets)
        return beads, fitted_params, results

    def _create_results_table(self, beads, fitted_params, img_name, offsets):
        c_image_name = []
        c_date = []
        c_microscope = []
        c_mag = []
        c_na = []
        c_x = []
        c_y = []
        c_z = []
        c_fwhm_x = []
        c_fwhm_y = []
        c_fwhm_z = []
        c_r2_x = []
        c_r2_y = []
        c_r2_z = []
        c_s2bg = []
        c_xyspacing = []
        c_zspacing = []
        for params, bead, offset in zip(fitted_params, beads, offsets):
            height = params[0]
            background = params[-1]
            mu_x = params[3]
            mu_y = params[2]
            mu_z = params[1]
            sigma_x = params[6]
            sigma_y = params[5]
            sigma_z = params[4]
            c_image_name.append(img_name)
            c_date.append(self.date.strftime("%Y-%m-%d"))
            c_microscope.append(self.microscope)
            c_mag.append(self.magnification)
            c_na.append(self.NA)
            c_x.append(mu_x + offset[2])
            c_y.append(mu_y + offset[1])
            c_z.append(mu_z + offset[0])
            c_fwhm_x.append(abs(self._fwhm(sigma_x)) * self.spacing[2])
            c_fwhm_y.append(abs(self._fwhm(sigma_y)) * self.spacing[1])
            c_fwhm_z.append(abs(self._fwhm(sigma_z)) * self.spacing[0])
            c_r2_x.append(
                self._r_squared(
                    bead[int(np.round(mu_z)), int(np.round(mu_y))],
                    height,
                    mu_x,
                    sigma_x,
                    background,
                )
            )
            c_r2_y.append(
                self._r_squared(
                    bead[int(np.round(mu_z)), :, int(np.round(mu_x))],
                    height,
                    mu_y,
                    sigma_y,
                    background,
                )
            )
            c_r2_z.append(
                self._r_squared(
                    bead[:, int(np.round(mu_y)), int(np.round(mu_x))],
                    height,
                    mu_z,
                    sigma_z,
                    background,
                )
            )
            c_s2bg.append(self._get_signal(bead, mu_z, mu_y, mu_x) / background)
            c_xyspacing.append(self.spacing[1])
            c_zspacing.append(self.spacing[0])
        results = pd.DataFrame(
            {
                "ImageName": c_image_name,
                "Date": c_date,
                "Microscope": c_microscope,
                "Magnification": c_mag,
                "NA": c_na,
                "X": c_x,
                "Y": c_y,
                "Z": c_z,
                "FWHM_X": c_fwhm_x,
                "FWHM_Y": c_fwhm_y,
                "FWHM_Z": c_fwhm_z,
                "r2_x": c_r2_x,
                "r2_y": c_r2_y,
                "r2_z": c_r2_z,
                "SignalToBG": c_s2bg,
                "XYpixelsize": c_xyspacing,
                "Zspacing": c_zspacing,
            }
        )
        return results
