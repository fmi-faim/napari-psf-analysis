from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy._typing import ArrayLike

from napari_psf_analysis.psf_analysis.extract.BeadExtractor import BeadExtractor
from napari_psf_analysis.psf_analysis.image import Calibrated3DImage
from napari_psf_analysis.psf_analysis.parameters import PSFAnalysisInputs
from napari_psf_analysis.psf_analysis.psf import PSF


class Analyzer:
    def __init__(self, parameters: PSFAnalysisInputs):
        self._parameters = parameters
        bead_extractor = BeadExtractor(
            image=Calibrated3DImage(
                data=parameters.img_data, spacing=parameters.spacing
            ),
            patch_size=parameters.patch_size,
        )
        self._beads = bead_extractor.extract_beads(points=self._parameters.point_data)

        self._results = None
        self._result_figures = {}
        self._index = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._beads)

    def __next__(self):
        if self._index < len(self._beads):
            bead = self._beads[self._index]
            psf = PSF(image=bead)

            psf.analyze()
            results = psf.get_summary_dict()
            self._add(
                self._extend_result_table(bead, results),
                psf.get_summary_image(
                    date=self._parameters.date,
                    version=self._parameters.version,
                    dpi=self._parameters.dpi,
                ),
            )
            self._index += 1
            return self._index
        else:
            raise StopIteration()

    def _extend_result_table(self, bead, results):
        extended_results = results.copy()
        extended_results["z_mu"] += bead.offset[0] * self._parameters.spacing[0]
        extended_results["y_mu"] += bead.offset[1] * self._parameters.spacing[1]
        extended_results["x_mu"] += bead.offset[2] * self._parameters.spacing[2]
        extended_results["image_name"] = self._parameters.name
        extended_results["date"] = self._parameters.date
        extended_results["microscope"] = self._parameters.microscope
        extended_results["mag"] = self._parameters.magnification
        extended_results["NA"] = self._parameters.na
        extended_results["yx_spacing"] = self._parameters.spacing[1]
        extended_results["z_spacing"] = self._parameters.spacing[0]
        extended_results["version"] = self._parameters.version
        return extended_results

    def _add(self, result: dict, summary_fig: ArrayLike):
        if self._results is None:
            self._results = {}
            for key in result.keys():
                self._results[key] = [result[key]]
        else:
            for key in result.keys():
                self._results[key].append(result[key])

        centroid = np.round([result["x_mu"], result["y_mu"], result["z_mu"]], 1)
        bead_name = "{}_Bead_X{}_Y{}_Z{}".format(result["image_name"], *centroid)

        self._result_figures[self._make_unique(bead_name)] = summary_fig

    def _make_unique(self, name: str):
        count = 1
        unique_name = name
        while unique_name in self._result_figures.keys():
            unique_name = f"{name}-{count}"
            count += 1
        return unique_name

    def get_results(self) -> Optional[pd.DataFrame]:
        """Create result table from dict.

        Parameters
        ----------
        results :
            Result dict obtained from `analyze_bead`

        Returns
        -------
        result_table
            Result table with "nice" column names
        """
        if self._results is not None:
            return self._build_dataframe()
        else:
            return None

    def get_summary_figure_stack(
        self,
        bead_img_scale: Tuple[float, float, float],
        bead_img_shape: Tuple[int, int, int],
    ) -> Optional[Tuple[ArrayLike, ArrayLike]]:
        """
        Create a (N, Y, X, 3) stack of all summary figures.

        Parameters
        ----------
        bead_img_scale : scaling of the whole bead image
        bead_img_shape : shape of the whole bead image

        Returns
        -------
        stack of all summary figures
        scaling to display them with napari
        """
        if len(self._result_figures) > 0:
            measurement_stack = self._build_figure_stack()

            measurement_scale = self._compute_figure_scaling(
                bead_img_scale, bead_img_shape, measurement_stack
            )
            return measurement_stack, measurement_scale
        else:
            return None, None

    def _compute_figure_scaling(self, bead_scale, bead_shape, measurement_stack):
        measurement_scale = np.array(
            [
                bead_scale[0],
                bead_scale[1] / measurement_stack.shape[1] * bead_shape[1],
                bead_scale[1] / measurement_stack.shape[1] * bead_shape[1],
            ]
        )
        return measurement_scale

    def _build_figure_stack(self):
        measurement_stack = np.stack(
            [self._result_figures[k] for k in self._result_figures.keys()]
        )
        return measurement_stack

    def _build_dataframe(self):
        return pd.DataFrame(
            {
                "ImageName": self._results["image_name"],
                "Date": self._results["date"],
                "Microscope": self._results["microscope"],
                "Magnification": self._results["mag"],
                "NA": self._results["NA"],
                "Amplitude_1D_Z": self._results["z_amp"],
                "Amplitude_2D_XY": self._results["yx_amp"],
                "Amplitude_3D_XYZ": self._results["zyx_amp"],
                "Background_1D_Z": self._results["z_bg"],
                "Background_2D_XY": self._results["yx_bg"],
                "Background_3D_XYZ": self._results["zyx_bg"],
                "Z_1D": self._results["z_mu"],
                "X_2D": self._results["x_mu"],
                "Y_2D": self._results["y_mu"],
                "X_3D": self._results["zyx_x_mu"],
                "Y_3D": self._results["zyx_y_mu"],
                "Z_3D": self._results["zyx_z_mu"],
                "FWHM_1D_Z": self._results["z_fwhm"],
                "FWHM_2D_X": self._results["x_fwhm"],
                "FWHM_2D_Y": self._results["y_fwhm"],
                "FWHM_3D_Z": self._results["zyx_z_fwhm"],
                "FWHM_3D_Y": self._results["zyx_y_fwhm"],
                "FWHM_3D_X": self._results["zyx_x_fwhm"],
                "FWHM_PA1_2D": self._results["yx_pc1_fwhm"],
                "FWHM_PA2_2D": self._results["yx_pc2_fwhm"],
                "FWHM_PA1_3D": self._results["zyx_pc1_fwhm"],
                "FWHM_PA2_3D": self._results["zyx_pc2_fwhm"],
                "FWHM_PA3_3D": self._results["zyx_pc3_fwhm"],
                "SignalToBG_1D_Z": (
                    np.array(self._results["z_amp"]) / np.array(self._results["z_bg"])
                ).tolist(),
                "SignalToBG_2D_XY": (
                    np.array(self._results["yx_amp"]) / np.array(self._results["yx_bg"])
                ).tolist(),
                "SignalToBG_3D_XYZ": (
                    np.array(self._results["zyx_amp"])
                    / np.array(self._results["zyx_bg"])
                ).tolist(),
                "XYpixelsize": self._results["yx_spacing"],
                "Zspacing": self._results["z_spacing"],
                "cov_xx_3D": self._results["zyx_cxx"],
                "cov_xy_3D": self._results["zyx_cyx"],
                "cov_xz_3D": self._results["zyx_czx"],
                "cov_yy_3D": self._results["zyx_cyy"],
                "cov_yz_3D": self._results["zyx_czy"],
                "cov_zz_3D": self._results["zyx_czz"],
                "cov_xx_2D": self._results["yx_cxx"],
                "cov_xy_2D": self._results["yx_cyx"],
                "cov_yy_2D": self._results["yx_cyy"],
                "sde_amp_1D_Z": self._results["z_amp_sde"],
                "sde_amp_2D_XY": self._results["yx_amp_sde"],
                "sde_amp_3D_XYZ": self._results["zyx_amp_sde"],
                "sde_background_1D_Z": self._results["z_bg_sde"],
                "sde_background_2D_XY": self._results["yx_bg_sde"],
                "sde_background_3D_XYZ": self._results["zyx_bg_sde"],
                "sde_Z_1D": self._results["z_mu_sde"],
                "sde_X_2D": self._results["x_mu_sde"],
                "sde_Y_2D": self._results["y_mu_sde"],
                "sde_X_3D": self._results["zyx_x_mu_sde"],
                "sde_Y_3D": self._results["zyx_y_mu_sde"],
                "sde_Z_3D": self._results["zyx_z_mu_sde"],
                "sde_cov_xx_3D": self._results["zyx_cxx_sde"],
                "sde_cov_xy_3D": self._results["zyx_cyx_sde"],
                "sde_cov_xz_3D": self._results["zyx_czx_sde"],
                "sde_cov_yy_3D": self._results["zyx_cyy_sde"],
                "sde_cov_yz_3D": self._results["zyx_czx_sde"],
                "sde_cov_zz_3D": self._results["zyx_czz_sde"],
                "sde_cov_xx_2D": self._results["yx_cxx_sde"],
                "sde_cov_xy_2D": self._results["yx_cyx_sde"],
                "sde_cov_yy_2D": self._results["yx_cyy_sde"],
                "version": self._results["version"],
            }
        )
