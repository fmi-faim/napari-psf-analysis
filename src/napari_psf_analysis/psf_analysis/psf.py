from copy import copy
from typing import Dict, Tuple

import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.mplot3d.axis3d import Axis
from numpy._typing import ArrayLike

from napari_psf_analysis.psf_analysis.fit.fitter import YXFitter, ZFitter, ZYXFitter
from napari_psf_analysis.psf_analysis.image import Calibrated2DImage, Calibrated3DImage
from napari_psf_analysis.psf_analysis.records import (
    PSFRecord,
    YXFitRecord,
    ZFitRecord,
    ZYXFitRecord,
)
from napari_psf_analysis.psf_analysis.sample import YXSample

# Patch matplotlib to render ticks in 3D correctly.
# See: https://stackoverflow.com/a/16496436
if not hasattr(Axis, "_get_coord_info_old"):

    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs

    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new


class PSFRenderEngine:
    from matplotlib.figure import Axes, Figure

    psf_image: Calibrated3DImage = None
    psf_record: PSFRecord = None
    _figure: Figure = None
    _ax_xy: Axes = None
    _ax_yz: Axes = None
    _ax_zx: Axes = None
    _ax_3d: Axes = None
    _ax_3d_text: Axes = None

    def __init__(self, psf_image: Calibrated3DImage, psf_record: PSFRecord):
        self.psf_image = psf_image
        self.psf_record = psf_record

    def _build_layout(self, dpi: int = 300) -> None:
        import matplotlib.pyplot as plt

        self._figure = plt.figure(figsize=(10, 10), dpi=dpi)
        self._add_axes()

    def _add_axes(self):
        self._ax_xy = self._figure.add_axes([0.025, 0.525, 0.45, 0.45])
        self._ax_yz = self._figure.add_axes([0.525, 0.525, 0.45, 0.45])
        self._ax_zx = self._figure.add_axes([0.025, 0.025, 0.45, 0.45])
        self._ax_3d = self._figure.add_axes(
            [0.55, 0.025, 0.42, 0.42], projection="3d", computed_zorder=False
        )
        self._ax_3d_text = self._figure.add_axes([0.525, 0.025, 0.45, 0.45])
        self._ax_3d_text.axis("off")

        self._add_axis_annotations()

    def _add_axis_annotations(self) -> None:
        ax_X = self._figure.add_axes([0.25, 0.5, 0.02, 0.02])
        ax_X.text(0, -0.1, "X", fontsize=14, ha="center", va="center")
        ax_X.axis("off")
        ax_Y = self._figure.add_axes([0.5, 0.75, 0.02, 0.02])
        ax_Y.text(0, 0, "Y", fontsize=14, ha="center", va="center")
        ax_Y.axis("off")
        ax_Z = self._figure.add_axes([0.5, 0.25, 0.02, 0.02])
        ax_Z.text(-0.5, 0, "Z", fontsize=14, ha="center", va="center")
        ax_Z.axis("off")
        ax_Z1 = self._figure.add_axes([0.75, 0.5, 0.02, 0.02])
        ax_Z1.text(0, 0.5, "Z", fontsize=14, ha="center", va="center")
        ax_Z1.axis("off")

    def render(
        self, date: str = None, version: str = None, dpi: int = 300
    ) -> ArrayLike:
        self._build_layout(dpi=dpi)
        self._add_projections(dpi=dpi)
        self._add_ellipsoids()

        if date is not None:
            self._add_date(date)

        if version is not None:
            self._add_version(version)

        return self._fig_to_image()

    def _add_projections(self, dpi: int):
        self._add_yx_projection(dpi=dpi)
        self._add_zx_projection(dpi=dpi)
        self._add_yz_projection(dpi=dpi)

    def _add_yz_projection(self, dpi: int):
        yz_sqrt_projection = self._compute_centered_isotropic_image(
            projection=Calibrated2DImage(
                data=np.sqrt(np.max(self.psf_image.data, axis=2)).T,
                spacing=(self.psf_image.spacing[1], self.psf_image.spacing[0]),
            ),
            centroid=(self.psf_record.yx_fit.y_mu, self.psf_record.z_fit.z_mu),
            dpi=dpi,
        )
        self._add_image_to_axes(yz_sqrt_projection.data, self._ax_yz, origin="lower")
        self._draw_fwhm_annotations(
            self._ax_yz,
            shape=yz_sqrt_projection.data.shape,
            spacing=yz_sqrt_projection.spacing,
            fwhm_values=(self.psf_record.yx_fit.y_fwhm, self.psf_record.z_fit.z_fwhm),
            origin_upper=False,
        )

    def _add_zx_projection(self, dpi: int):
        zx_sqrt_projection = self._compute_centered_isotropic_image(
            projection=Calibrated2DImage(
                data=np.sqrt(np.max(self.psf_image.data, axis=1)),
                spacing=(self.psf_image.spacing[0], self.psf_image.spacing[2]),
            ),
            centroid=(self.psf_record.z_fit.z_mu, self.psf_record.yx_fit.x_mu),
            dpi=dpi,
        )
        self._add_image_to_axes(zx_sqrt_projection.data, self._ax_zx, origin="upper")
        self._draw_fwhm_annotations(
            self._ax_zx,
            shape=zx_sqrt_projection.data.shape,
            spacing=zx_sqrt_projection.spacing,
            fwhm_values=(
                self.psf_record.z_fit.z_fwhm,
                self.psf_record.yx_fit.x_fwhm,
            ),
            origin_upper=True,
        )

    def _add_yx_projection(self, dpi: int):
        yx_sqrt_projection = self._compute_centered_isotropic_image(
            projection=Calibrated2DImage(
                data=np.sqrt(np.max(self.psf_image.data, axis=0)),
                spacing=self.psf_image.spacing[1:],
            ),
            centroid=(self.psf_record.yx_fit.y_mu, self.psf_record.yx_fit.x_mu),
            dpi=dpi,
        )
        self._add_image_to_axes(yx_sqrt_projection.data, self._ax_xy, origin="lower")
        self._add_scalebar(yx_sqrt_projection.data.shape)
        self._draw_fwhm_annotations(
            self._ax_xy,
            shape=yx_sqrt_projection.data.shape,
            spacing=yx_sqrt_projection.spacing,
            fwhm_values=(
                self.psf_record.yx_fit.y_fwhm,
                self.psf_record.yx_fit.x_fwhm,
            ),
            origin_upper=False,
        )

    def _get_display_min_max(self, image: ArrayLike) -> Dict[str, float]:
        return {
            "vmin": np.quantile(image[image != -1], 0.03),
            "vmax": image[image != -1].max(),
        }

    def _add_image_to_axes(self, image: ArrayLike, ax: Axes, origin: str):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(
            self._num_to_nan(image, num=-1),
            cmap=self._get_cmap(),
            interpolation="nearest",
            origin=origin,
            **self._get_display_min_max(image),
        )

    def _add_scalebar(self, shape: Tuple[int, int]):
        scalebar = ScaleBar(
            4000 / shape[1],
            "nm",
            fixed_value=500,
            location="lower right",
            border_pad=0.2,
            box_alpha=0.8,
        )
        self._ax_xy.add_artist(scalebar)

    def _get_cmap(self):
        from matplotlib import cm

        cmap = copy(cm.turbo)
        cmap.set_bad("white")
        return cmap

    def _num_to_nan(self, data: ArrayLike, num: float = -1):
        replaced = copy(data)
        replaced[replaced == num] = np.nan
        return replaced

    def _compute_centered_isotropic_image(
        self, projection: Calibrated2DImage, centroid: Tuple[float, float], dpi: int
    ) -> Calibrated2DImage:
        padded_projection = YXSample(
            image=Calibrated2DImage(
                data=np.pad(projection.data, 1, mode="constant", constant_values=(-1,)),
                spacing=projection.spacing,
            )
        )

        source_points = (
            padded_projection.get_ravelled_coordinates()
            - np.array([centroid])
            - np.array(padded_projection.image.spacing)
        )

        num_samples = self._estimate_number_of_samples(
            spacing=padded_projection.image.spacing,
            dpi=dpi,
        )
        target_points = self._get_target_points(num_samples=num_samples)

        import scipy

        return Calibrated2DImage(
            data=scipy.interpolate.griddata(
                points=source_points,
                values=padded_projection.image.data.ravel(),
                xi=target_points,
                method="nearest",
            ).reshape(num_samples, num_samples),
            spacing=(4000 / num_samples,) * 2,
        )

    def _get_target_points(self, num_samples: int):
        yy, xx = np.meshgrid(
            np.arange(num_samples) * 4000 / num_samples - 2000,
            np.arange(num_samples) * 4000 / num_samples - 2000,
            indexing="ij",
        )
        return np.stack([yy.ravel(), xx.ravel()], axis=-1)

    def _estimate_number_of_samples(self, spacing: Tuple[float, float], dpi: int):
        num_samples = int(np.min(spacing) * 10)
        while num_samples < 4.5 * dpi:
            num_samples += int(np.min(spacing) * 10)

        return num_samples

    def _draw_fwhm_annotations(
        self,
        axes: Axes,
        shape: Tuple[int, int],
        spacing: Tuple[float, float],
        fwhm_values: Tuple[float, float],
        origin_upper: bool,
    ) -> None:
        self._add_vertical_fwhm_annotation(
            axes=axes,
            fwhm_values=fwhm_values,
            shape=shape,
            spacing=spacing,
        )

        self._add_horizontal_fwhm_annotation(
            axes=axes,
            fwhm_values=fwhm_values,
            shape=shape,
            spacing=spacing,
            origin_upper=origin_upper,
        )

    def _add_horizontal_fwhm_annotation(
        self,
        axes: Axes,
        fwhm_values: Tuple[float, float],
        shape: Tuple[int, int],
        spacing: Tuple[float, float],
        origin_upper: bool,
    ):
        if origin_upper:
            shift_factor = 3
            start_factor = 1
            text_pos = shape[1] - shape[1] / 4.5
        else:
            shift_factor = 1
            start_factor = -1
            text_pos = shape[1] / 4.5

        cy = shape[0] / 2
        cx = shape[1] / 2
        y_fwhm, x_fwhm = fwhm_values

        if not np.isnan(y_fwhm) and not np.isnan(x_fwhm):
            dy = (y_fwhm / 2) / spacing[0]
            dx = (x_fwhm / 2) / spacing[1]

            axes.plot(
                [cx - dx, cx + dx],
                [
                    shift_factor * shape[1] / 4,
                ]
                * 2,
                linewidth=6,
                c="black",
                solid_capstyle="butt",
            )
            axes.plot(
                [cx - dx, cx + dx],
                [
                    shift_factor * shape[1] / 4,
                ]
                * 2,
                linewidth=4,
                c="white",
                solid_capstyle="butt",
            )
            axes.plot(
                [cx - dx, cx - dx],
                [cy + start_factor * dy, shift_factor * shape[1] / 4],
                "-",
                c="black",
            )
            axes.plot(
                [cx + dx, cx + dx],
                [cy + start_factor * dy, shift_factor * shape[1] / 4],
                "-",
                c="black",
            )
            axes.plot(
                [cx - dx, cx - dx],
                [cy + start_factor * dy, shift_factor * shape[1] / 4],
                "--",
                c="white",
            )
            axes.plot(
                [cx + dx, cx + dx],
                [cy + start_factor * dy, shift_factor * shape[1] / 4],
                "--",
                c="white",
            )

        axes.text(
            cx,
            text_pos,
            f"{self._get_fwhm_str(x_fwhm)}nm",
            ha="center",
            va="top",
            fontsize=15,
            bbox=dict(facecolor="white", alpha=0.8, linewidth=0),
        )

    def _get_fwhm_str(self, fwhm: float):
        if np.isnan(fwhm):
            return "NaN"
        else:
            return f"{int(np.round(fwhm))}"

    def _add_vertical_fwhm_annotation(
        self,
        axes: Axes,
        fwhm_values: Tuple[float, float],
        shape: Tuple[int, int],
        spacing: Tuple[float, float],
    ):
        cy = shape[0] / 2
        cx = shape[1] / 2
        y_fwhm, x_fwhm = fwhm_values

        if not np.isnan(y_fwhm) and not np.isnan(x_fwhm):
            dy = (y_fwhm / 2) / spacing[0]
            dx = (x_fwhm / 2) / spacing[1]

            axes.plot(
                [
                    shape[0] / 4,
                ]
                * 2,
                [cy - dy, cy + dy],
                linewidth=6,
                c="black",
                solid_capstyle="butt",
            )
            axes.plot(
                [
                    shape[0] / 4,
                ]
                * 2,
                [cy - dy, cy + dy],
                linewidth=4,
                c="white",
                solid_capstyle="butt",
            )
            axes.plot([shape[0] / 4, cx - dx], [cy - dy, cy - dy], "-", color="black")
            axes.plot([shape[0] / 4, cx - dx], [cy + dy, cy + dy], "-", color="black")
            axes.plot([shape[0] / 4, cx - dx], [cy - dy, cy - dy], "--", color="white")
            axes.plot([shape[0] / 4, cx - dx], [cy + dy, cy + dy], "--", color="white")

        axes.text(
            shape[1] / 4.5,
            cy,
            f"{self._get_fwhm_str(y_fwhm)}nm",
            ha="right",
            va="center",
            fontsize=15,
            bbox=dict(facecolor="white", alpha=0.8, linewidth=0),
        )

    def _has_nan_fwhm(self) -> bool:
        if np.isnan(self.psf_record.z_fit.z_fwhm):
            return True

        if np.isnan(self.psf_record.yx_fit.y_fwhm):
            return True

        if np.isnan(self.psf_record.yx_fit.x_fwhm):
            return True

        return False

    def _add_ellipsoids(self):
        self._configure_ticks_and_bounds()

        if not self._has_nan_fwhm():
            self._add_axis_aligned_ellipsoid()

        self._add_zyx_ellipsoid()
        self._add_principal_components_annotons()

    def _add_axis_aligned_ellipsoid(self):
        from napari_psf_analysis.psf_analysis.utils import sigma

        covariance = np.array(
            [
                [sigma(self.psf_record.yx_fit.x_fwhm) ** 2, 0, 0],
                [0, sigma(self.psf_record.yx_fit.y_fwhm) ** 2, 0],
                [0, 0, sigma(self.psf_record.z_fit.z_fwhm) ** 2],
            ]
        )
        base_ell = self._get_ellipsoid(
            covariance=covariance,
            spacing=self.psf_image.spacing,
        )
        self._ax_3d.plot_surface(
            *base_ell, rstride=2, cstride=2, color="white", antialiased=True, alpha=1
        )
        self._ax_3d.plot_wireframe(
            *base_ell, rstride=3, cstride=3, color="black", antialiased=True, alpha=0.5
        )

        self._ax_3d.contour(
            *base_ell,
            zdir="z",
            offset=self._ax_3d.get_zlim()[0],
            colors="white",
            levels=1,
            vmin=-1,
            vmax=1,
            zorder=0,
        )
        self._ax_3d.contour(
            *base_ell,
            zdir="y",
            offset=self._ax_3d.get_ylim()[1],
            colors="white",
            levels=1,
            vmin=-1,
            vmax=1,
            zorder=0,
        )
        self._ax_3d.contour(
            *base_ell,
            zdir="x",
            offset=self._ax_3d.get_xlim()[0],
            colors="white",
            levels=1,
            vmin=-1,
            vmax=1,
            zorder=0,
        )
        self._ax_3d.contour(
            *base_ell,
            zdir="z",
            offset=self._ax_3d.get_zlim()[0],
            colors="black",
            levels=1,
            vmin=-1,
            vmax=1,
            zorder=0,
            linestyles="--",
        )
        self._ax_3d.contour(
            *base_ell,
            zdir="y",
            offset=self._ax_3d.get_ylim()[1],
            colors="black",
            levels=1,
            vmin=-1,
            vmax=1,
            zorder=0,
            linestyles="--",
        )
        self._ax_3d.contour(
            *base_ell,
            zdir="x",
            offset=self._ax_3d.get_xlim()[0],
            colors="black",
            levels=1,
            vmin=-1,
            vmax=1,
            zorder=0,
            linestyles="--",
        )

    def _get_ellipsoid(
        self, covariance: ArrayLike, spacing: Tuple[float, float, float]
    ):
        bias = 0
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        cov_ = covariance / 2 * np.sqrt(2 * np.log(2))
        x, y, z = (cov_ @ np.stack((x, y, z), 0).reshape(3, -1) + bias).reshape(
            3, *x.shape
        )
        return x / spacing[2], y / spacing[1], z / spacing[0]

    def _add_zyx_ellipsoid(self):
        fit = self.psf_record.zyx_fit
        covariance = np.array(
            [
                [fit.zyx_cxx, fit.zyx_cyx, fit.zyx_czx],
                [fit.zyx_cyx, fit.zyx_cyy, fit.zyx_czy],
                [fit.zyx_czx, fit.zyx_czy, fit.zyx_czz],
            ]
        )
        cv_ell = self._get_ellipsoid(
            covariance=covariance, spacing=self.psf_image.spacing
        )
        self._ax_3d.plot_surface(
            *cv_ell, rstride=1, cstride=1, color="navy", antialiased=True, alpha=0.15
        )
        self._ax_3d.plot_wireframe(
            *cv_ell, rstride=4, cstride=4, color="navy", antialiased=True, alpha=0.25
        )

        self._ax_3d.contour(
            *cv_ell,
            zdir="z",
            offset=self._ax_3d.get_zlim()[0],
            colors="navy",
            levels=1,
            vmin=-1,
            vmax=1,
            zorder=0,
        )
        self._ax_3d.contour(
            *cv_ell,
            zdir="y",
            offset=self._ax_3d.get_ylim()[1],
            colors="navy",
            levels=1,
            vmin=-1,
            vmax=1,
            zorder=0,
        )
        self._ax_3d.contour(
            *cv_ell,
            zdir="x",
            offset=self._ax_3d.get_xlim()[0],
            colors="navy",
            levels=1,
            vmin=-1,
            vmax=1,
            zorder=0,
        )

    def _configure_ticks_and_bounds(self):
        self._ax_3d.set_box_aspect((1.0, 1.0, 4.0 / 3.0))
        self._ax_3d.set_xticks([-1000, 0, 1000])
        self._ax_3d.set_xticklabels(
            [-1000, 0, 1000], verticalalignment="bottom", horizontalalignment="right"
        )
        self._ax_3d.set_yticks([-1000, 0, 1000])
        self._ax_3d.set_yticklabels(
            [-1000, 0, 1000], verticalalignment="bottom", horizontalalignment="left"
        )
        self._ax_3d.set_zticks([-2000, -1000, 0, 1000, 2000])
        self._ax_3d.set_zticklabels(
            [-2000, -1000, 0, 1000, 2000],
            verticalalignment="top",
            horizontalalignment="left",
        )
        self._ax_3d.set_xlim(-1500, 1500)
        self._ax_3d.set_ylim(-1500, 1500)
        self._ax_3d.set_zlim(-2000, 2000)

    def _add_principal_components_annotons(self):
        self._ax_3d_text.set_xlim(0, 100)
        self._ax_3d_text.set_ylim(0, 100)
        self._ax_3d_text.text(
            3,
            95,
            "Principal Axes",
            fontsize=14,
            weight="bold",
            color="navy",
        )
        self._ax_3d_text.text(
            5,
            89,
            f"{self._get_fwhm_str(self.psf_record.zyx_fit.zyx_pc1_fwhm)}nm",
            fontsize=14,
            color="navy",
        )
        self._ax_3d_text.text(
            5,
            83,
            f"{self._get_fwhm_str(self.psf_record.zyx_fit.zyx_pc2_fwhm)}nm",
            fontsize=14,
            color="navy",
        )
        self._ax_3d_text.text(
            5,
            77,
            f"{self._get_fwhm_str(self.psf_record.zyx_fit.zyx_pc2_fwhm)}nm",
            fontsize=14,
            color="navy",
        )

    def _add_date(self, date: str) -> None:
        self._ax_3d_text.text(
            100,
            -4,
            f"Acquisition date: {date}",
            fontsize=11,
            horizontalalignment="right",
        )

    def _add_version(self, version: str) -> None:
        self._ax_3d_text.text(-110, -4, f"napari-psf-analysis: v{version}", fontsize=11)

    def _fig_to_image(self):
        from matplotlib_inline.backend_inline import FigureCanvas

        canvas = FigureCanvas(self._figure)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
            (self._figure.canvas.get_width_height()[::-1]) + (3,)
        )
        from matplotlib import pyplot as plt

        plt.close(self._figure)
        return image


class PSF:
    image: Calibrated3DImage = None
    psf_record: PSFRecord = None

    def __init__(self, image: Calibrated3DImage):
        self.image = image

    def analyze(self) -> None:
        z_fitter = ZFitter(image=self.image)
        yx_fitter = YXFitter(image=self.image)
        zyx_fitter = ZYXFitter(image=self.image)

        z_fit_record: ZFitRecord = z_fitter.fit()
        yx_fit_record: YXFitRecord = yx_fitter.fit()
        zyx_fit_record: ZYXFitRecord = zyx_fitter.fit()

        self.psf_record = PSFRecord(
            z_fit=z_fit_record,
            yx_fit=yx_fit_record,
            zyx_fit=zyx_fit_record,
        )

    def get_record(self) -> PSFRecord:
        return self.psf_record

    def get_summary_image(
        self,
        date: str = None,
        version: str = None,
        dpi: int = 300,
    ) -> ArrayLike:
        engine = PSFRenderEngine(psf_image=self.image, psf_record=self.psf_record)
        return engine.render(date=date, version=version, dpi=dpi)

    def get_summary_dict(self) -> dict:
        return {
            **self.psf_record.z_fit.dict(),
            **self.psf_record.yx_fit.dict(),
            **self.psf_record.zyx_fit.dict(),
        }
