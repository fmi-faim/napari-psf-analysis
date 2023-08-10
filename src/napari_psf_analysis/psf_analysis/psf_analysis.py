from copy import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.interpolate
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib_inline.backend_inline import FigureCanvas
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.mplot3d.axis3d import Axis
from napari.utils.notifications import show_info
from numpy._typing import ArrayLike
from scipy.optimize import curve_fit
from skimage.filters import gaussian

from napari_psf_analysis.psf_analysis.fit_1d import evaluate_1d_gaussian
from napari_psf_analysis.psf_analysis.fit_2d import evaluate_2d_gaussian
from napari_psf_analysis.psf_analysis.fit_3d import evaluate_3d_gaussian
from napari_psf_analysis.psf_analysis.utils import estimate_from_data, fwhm

if not hasattr(Axis, "_get_coord_info_old"):

    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs

    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new


def fit_1d_z(
    img: ArrayLike,
    spacing: Tuple[float, float, float],
) -> Tuple[ArrayLike, ArrayLike]:
    """Fit 1D Gaussian along Z axis.

    It is assumed that the bead is centered i.e. the maximum intensity value is
    in the voxel
    img[img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2].

    Parameters
    ----------
    img :
        3D image of a spot
    spacing :
        Voxel spacing

    Returns
    -------
    popt:
        Fitted values for background, amplitude, mu and sigma
    perr:
        The corresponding standard deviation errors.
    """
    assert img.ndim == 3, "Bead image must have 3 dimensions."

    z_sample = img[:, img.shape[1] // 2, img.shape[2] // 2]
    z_coords = np.arange(z_sample.shape[0]) * spacing[0]

    bg, amp, mu, sigma = estimate_from_data(
        sample=z_sample, data=img, sample_spacing=spacing[:1]
    )

    popt, pcov = curve_fit(
        evaluate_1d_gaussian, z_coords, ydata=z_sample, p0=[bg, amp, mu[0], sigma]
    )

    perr = np.sqrt(np.diag(pcov))

    return popt, perr


def fit_2d_yx(
    img: ArrayLike, spacing: Tuple[float, float, float]
) -> Tuple[ArrayLike, ArrayLike]:
    """Fit 2D Gaussian in YX plane.

    It is assumed that the bead is centered i.e. the maximum intensity value is
    in the voxel
    img[img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2].

    Parameters
    ----------
    img :
        3D image of a spot
    spacing :
        Voxel spacing

    Returns
    -------
    popt:
        Fitted values for background, amplitude, mu and sigma
    perr:
        The corresponding standard deviation errors.
    """
    assert img.ndim == 3, "Bead image must have 3 dimensions."

    yx_sample = img[img.shape[0] // 2]

    yy = np.arange(yx_sample.shape[0]) * spacing[1]
    xx = np.arange(yx_sample.shape[1]) * spacing[2]
    y, x = np.meshgrid(yy, xx, indexing="ij")
    yx_coords = np.stack([y.ravel(), x.ravel()], -1)

    bg, amp, mus, sigmas = estimate_from_data(
        sample=yx_sample, data=img, sample_spacing=spacing[1:]
    )

    popt, pcov = curve_fit(
        evaluate_2d_gaussian,
        yx_coords,
        ydata=yx_sample.ravel(),
        p0=[bg, amp, mus[0], mus[1], sigmas[0] ** 2, 0, sigmas[1] ** 2],
    )

    perr = np.sqrt(np.diag(pcov))

    return popt, perr


def fit_3d_zyx(
    img: ArrayLike, spacing: Tuple[float, float, float]
) -> Tuple[ArrayLike, ArrayLike]:
    """Fit 3D Gaussian.

    It is assumed that the bead is centered i.e. the maximum intensity value is
    in the voxel
    img[img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2].

    Parameters
    ----------
    img :
        3D image of a spot
    spacing :
        Voxel spacing

    Returns
    -------
    popt:
        Fitted values for background, amplitude, mu and sigma
    perr:
        The corresponding standard deviation errors.
    """
    assert img.ndim == 3, "Bead image must have 3 dimensions."

    zz = np.arange(img.shape[0]) * spacing[0]
    yy = np.arange(img.shape[1]) * spacing[1]
    xx = np.arange(img.shape[2]) * spacing[2]
    z, y, x = np.meshgrid(zz, yy, xx, indexing="ij")
    zyx_coords = np.stack([z.ravel(), y.ravel(), x.ravel()], -1)

    bg, amp, mus, sigmas = estimate_from_data(
        sample=img, data=img, sample_spacing=spacing
    )

    popt, pcov = curve_fit(
        evaluate_3d_gaussian,
        zyx_coords,
        ydata=img.ravel(),
        p0=[
            bg,
            amp,
            mus[0],
            mus[1],
            mus[2],
            sigmas[0] ** 2,
            0,
            0,
            sigmas[1] ** 2,
            0,
            sigmas[2] ** 2,
        ],
    )

    perr = np.sqrt(np.diag(pcov))

    return popt, perr


def analyze_bead(
    bead: ArrayLike,
    spacing: Tuple[float, float, float],
) -> Dict:
    """Analyze bead image by fitting Gaussians.

    The bead image is characterized by fitting
    - a 1D Gaussian along the Z-axis passing through the center voxel of the
      image,
    - a 2D Gaussian to the YX plane passing through the center voxel of the
      image,
    - and a 3D Gaussian to the whole image
    with a least squares minimization strategy (see scipy.optimize.curve_fit).
    The 2D and 3D Gaussian fits can be rotated arbitrarily.

    It is assumed that the brightest pixel is the center voxel of the bead
    image.

    Parameters
    ----------
    bead :
        image of a centered bead
    spacing :
        spacing of the voxel data

    Returns
    -------
    estimated_parameters
        A dictionary with all estimated parameters and standard deviation
        errors:
        z_bg: Background of the 1D fit in Z
        z_amp: Amplitude of the 1D fit in Z
        z_mu: Subpixel localization of the bead in Z
        z_sigma: Sigma of the Gaussian fit in Z
        yx_bg: Background of the 2D fit in YX
        yx_amp: Amplitude of the 2D fit in YX
        y_mu: Subpixel localization of the bead in Y
        x_mu: Subpixel localization of the bead in X
        yx_cyy: Covariance value cyy of the 2D fit in YX
        yx_cyx: Covariance value cyx of the 2D fit in YX
        yx_cxx: Covariance value cxx of the 2D fit in YX
        z_fwhm: Full width half maximum in Z of 1D fit in Z
        y_fwhm: Full width half maximum in Y of 2D fit in YX
        x_fwhm: Full width half maximum in X of 2D fit in YX
        zyx_bg: Background of the 3D fit
        zyx_amp: Amplitude of the 3D fit
        zyx_z_mu: Subpixel localization in Z of the 3D fit
        zyx_y_mu: Subpixel localization in Y of the 3D fit
        zyx_x_mu: Subpixel localization in X of the 3D fit
        zyx_czz: Covariance value czz of the 3D fit in ZYX
        zyx_czy: Covariance value czy of the 3D fit in ZYX
        zyx_czx: Covariance value czx of the 3D fit in ZYX
        zyx_cyy: Covariance value cyy of the 3D fit in ZYX
        zyx_cyx: Covariance value cyx of the 3D fit in ZYX
        zyx_cxx: Covariance value cxx of the 3D fit in ZYX
        zyx_z_fwhm: Full width half maximum in Z of the 3D fit
        zyx_y_fwhm: Full width half maximum in Y of the 3D fit
        zyx_x_fwhm: Full width half maximum in X of the 3D fit
        zyx_pc1_fwhm: Full width half maximum of the 1st principal component
                      of the 3D fit
        zyx_pc2_fwhm: Full width half maximum of the 2nd principal component
                      of the 3D fit
        zyx_pc3_fwhm: Full width half maximum of the 3rd principal component
                      of the 3D fit
        z_bg_sde: One standard deviation error of z_bg
        z_amp_sde: One standard deviation error of z_amp
        z_mu_sde: One standard deviation error of z_mu
        z_sigma_sde: One standard deviation error of z_sigma
        yx_bg_sde: One standard deviation error of yx_bg
        yx_amp_sde: One standard deviation error of yb_amp
        y_mu_sde: One standard deviation error of y_mu
        x_mu_sde: One standard deviation error of x_mu
        yx_cyy_sde: One standard deviation error of yx_cyy
        yx_cyx_sde: One standard deviation error of yx_cyx
        yx_cxx_sde: One standard deviation error of yx_cxx
        zyx_bg_sde: One standard deviation error of zyx_bg
        zyx_amp_sde: One standard deviation error of zyx_amp
        zyx_z_mu_sde: One standard deviation error of zyx_z_mu
        zyx_y_mu_sde: One standard deviation error of zyx_y_mu
        zyx_x_mu_sde: One standard deviation error of zyx_x_mu
        zyx_czz_sde: One standard deviation error of zyx_czz
        zyx_czy_sde: One standard deviation error of zyx_czy
        zyx_czx_sde: One standard deviation error of zyx_czx
        zyx_cyy_sde: One standard deviation error of zyx_cyy
        zyx_cyx_sde: One standard deviation error of zyx_cyx
        zyx_cxx_sde: One standard deviation error of zyx_cxx
    """
    z_popt, z_perr = fit_1d_z(img=bead, spacing=spacing)

    yx_popt, yx_perr = fit_2d_yx(img=bead, spacing=spacing)

    zyx_popt, zyx_perr = fit_3d_zyx(img=bead, spacing=spacing)

    zyx_cov_matrix = np.array(
        [
            [zyx_popt[5], zyx_popt[6], zyx_popt[7]],
            [zyx_popt[6], zyx_popt[8], zyx_popt[9]],
            [zyx_popt[7], zyx_popt[9], zyx_popt[10]],
        ]
    )
    pc = np.sort(np.sqrt(np.linalg.eigvals(zyx_cov_matrix)))[::-1]

    return dict(
        z_bg=z_popt[0],
        z_amp=z_popt[1],
        z_mu=z_popt[2],
        z_sigma=z_popt[3],
        yx_bg=yx_popt[0],
        yx_amp=yx_popt[1],
        y_mu=yx_popt[2],
        x_mu=yx_popt[3],
        yx_cyy=yx_popt[4],
        yx_cyx=yx_popt[5],
        yx_cxx=yx_popt[6],
        z_fwhm=fwhm(z_popt[3]),
        y_fwhm=fwhm(np.sqrt(yx_popt[4])),
        x_fwhm=fwhm(np.sqrt(yx_popt[6])),
        zyx_bg=zyx_popt[0],
        zyx_amp=zyx_popt[1],
        zyx_z_mu=zyx_popt[2],
        zyx_y_mu=zyx_popt[3],
        zyx_x_mu=zyx_popt[4],
        zyx_czz=zyx_popt[5],
        zyx_czy=zyx_popt[6],
        zyx_czx=zyx_popt[7],
        zyx_cyy=zyx_popt[8],
        zyx_cyx=zyx_popt[9],
        zyx_cxx=zyx_popt[10],
        zyx_z_fwhm=fwhm(np.sqrt(zyx_popt[5])),
        zyx_y_fwhm=fwhm(np.sqrt(zyx_popt[8])),
        zyx_x_fwhm=fwhm(np.sqrt(zyx_popt[10])),
        zyx_pc1_fwhm=fwhm(pc[0]),
        zyx_pc2_fwhm=fwhm(pc[1]),
        zyx_pc3_fwhm=fwhm(pc[2]),
        z_bg_sde=z_perr[0],
        z_amp_sde=z_perr[1],
        z_mu_sde=z_perr[2],
        z_sigma_sde=z_perr[3],
        yx_bg_sde=yx_perr[0],
        yx_amp_sde=yx_perr[1],
        y_mu_sde=yx_perr[2],
        x_mu_sde=yx_perr[3],
        yx_cyy_sde=yx_perr[4],
        yx_cyx_sde=yx_perr[5],
        yx_cxx_sde=yx_perr[6],
        zyx_bg_sde=zyx_perr[0],
        zyx_amp_sde=zyx_perr[1],
        zyx_z_mu_sde=zyx_perr[2],
        zyx_y_mu_sde=zyx_perr[3],
        zyx_x_mu_sde=zyx_perr[4],
        zyx_czz_sde=zyx_perr[5],
        zyx_czy_sde=zyx_perr[6],
        zyx_czx_sde=zyx_perr[7],
        zyx_cyy_sde=zyx_perr[8],
        zyx_cyx_sde=zyx_perr[9],
        zyx_cxx_sde=zyx_perr[10],
    )


def _num_to_nan(data: ArrayLike, num: float = -1):
    data[data == num] = np.nan
    return data


def _draw_fwhm(
    axes,
    shape: Tuple[int, int],
    spacing: float,
    fwhm: Tuple[float, float],
    down: bool = False,
):
    cy = shape[0] / 2
    cx = shape[1] / 2
    y_fwhm, x_fwhm = fwhm
    dx = (x_fwhm / 2) / spacing
    dy = (y_fwhm / 2) / spacing

    if down:
        axes.plot(
            [cx - dx, cx + dx],
            [
                3 * shape[1] / 4,
            ]
            * 2,
            linewidth=6,
            c="black",
            solid_capstyle="butt",
        )
        axes.plot(
            [cx - dx, cx + dx],
            [
                3 * shape[1] / 4,
            ]
            * 2,
            linewidth=4,
            c="white",
            solid_capstyle="butt",
        )
        axes.plot([cx - dx, cx - dx], [cy + dy, 3 * shape[1] / 4], "-", c="black")
        axes.plot([cx + dx, cx + dx], [cy + dy, 3 * shape[1] / 4], "-", c="black")
        axes.plot([cx - dx, cx - dx], [cy + dy, 3 * shape[1] / 4], "--", c="white")
        axes.plot([cx + dx, cx + dx], [cy + dy, 3 * shape[1] / 4], "--", c="white")
        axes.text(
            cx,
            shape[1] - shape[1] / 4.5,
            f"{int(np.round(x_fwhm))}nm",
            ha="center",
            va="top",
            fontsize=15,
            bbox=dict(facecolor="white", alpha=0.8, linewidth=0),
        )
    else:
        axes.plot(
            [cx - dx, cx + dx],
            [
                shape[1] / 4,
            ]
            * 2,
            linewidth=6,
            c="black",
            solid_capstyle="butt",
        )
        axes.plot(
            [cx - dx, cx + dx],
            [
                shape[1] / 4,
            ]
            * 2,
            linewidth=4,
            c="white",
            solid_capstyle="butt",
        )
        axes.plot([cx - dx, cx - dx], [cy - dy, shape[1] / 4], "-", c="black")
        axes.plot([cx + dx, cx + dx], [cy - dy, shape[1] / 4], "-", c="black")
        axes.plot([cx - dx, cx - dx], [cy - dy, shape[1] / 4], "--", c="white")
        axes.plot([cx + dx, cx + dx], [cy - dy, shape[1] / 4], "--", c="white")
        axes.text(
            cx,
            shape[1] / 4.5,
            f"{int(np.round(x_fwhm))}nm",
            ha="center",
            va="top",
            fontsize=15,
            bbox=dict(facecolor="white", alpha=0.8, linewidth=0),
        )

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
        f"{int(np.round(y_fwhm))}nm",
        ha="right",
        va="center",
        fontsize=15,
        bbox=dict(facecolor="white", alpha=0.8, linewidth=0),
    )


def _get_ellipsoid(cov, spacing):
    bias = 0
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    cov_ = np.array(cov) / 2 * np.sqrt(2 * np.log(2))
    x, y, z = (cov_ @ np.stack((x, y, z), 0).reshape(3, -1) + bias).reshape(3, *x.shape)
    return x / spacing[2], y / spacing[1], z / spacing[0]


def build_summary_figure(
    bead_img: ArrayLike,
    spacing: Tuple[float, float, float],
    location: Tuple[float, float, float],
    fwhm_values: Tuple[float, float, float],
    cov_matrix_3d: ArrayLike,
    date: str = None,
    version: str = None,
) -> ArrayLike:
    """Build summary figure of PSF measurement.

    Build a figure summarizing the characterized point spread function of
    the bead image.

    Parameters
    ----------
    bead_img :
        The bead image of the measured point spread function.
    spacing :
        The voxel spacing of the bead image in nm.
    location :
        The subpixel localization of the bead in `bead_img` in nm.
    fwhm_values :
        Values of the computed full-width half maximum for X, Y,
        and Z dimensions.
    cov_matrix_3d :
        Estimated covariance matrix of the 3D Gaussian.
    date : Optional
        Acquisition date of the bead.
    version : Optional
        Version number of the plugin.

    Returns
    -------
    figure_image
        The figure as RGB image
    """
    fig = plt.figure(figsize=(10, 10))

    ax_xy, ax_yz, ax_zx, ax_3d, ax_3d_text = _add_axes(fig)

    yx_sqrt_projection = _to_iso_plane(
        plane=np.sqrt(np.max(bead_img, axis=0)),
        spacing=spacing[1:],
        location=location[1:],
    )
    vmin = np.quantile(yx_sqrt_projection[yx_sqrt_projection != -1], 0.03)
    vmax = yx_sqrt_projection[yx_sqrt_projection != -1].max()
    yx_sqrt_projection = _num_to_nan(yx_sqrt_projection)
    from matplotlib import cm

    cmap = copy(cm.turbo)
    cmap.set_bad("white")
    scalebar = ScaleBar(
        4000 / yx_sqrt_projection.shape[1],
        "nm",
        fixed_value=500,
        location="lower right",
        border_pad=0.2,
        box_alpha=0.8,
    )

    ax_xy.add_artist(scalebar)
    ax_xy.set_xticks([])
    ax_xy.set_yticks([])
    ax_xy.imshow(
        yx_sqrt_projection,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        origin="lower",
    )

    _draw_fwhm(
        ax_xy,
        spacing=4000 / yx_sqrt_projection.shape[1],
        fwhm=fwhm_values[1:],
        shape=yx_sqrt_projection.shape,
    )

    ax_zx.set_xticks([])
    ax_zx.set_yticks([])
    zx_plane = _to_iso_plane(
        plane=np.sqrt(np.max(bead_img, axis=1)),
        spacing=(spacing[0], spacing[2]),
        location=(location[0], location[2]),
    )
    ax_zx.imshow(
        _num_to_nan(zx_plane),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        origin="upper",
    )

    _draw_fwhm(
        ax_zx,
        spacing=4000 / zx_plane.shape[1],
        fwhm=(fwhm_values[0], fwhm_values[2]),
        shape=yx_sqrt_projection.shape,
        down=True,
    )

    ax_yz.set_xticks([])
    ax_yz.set_yticks([])
    yz_plane = _to_iso_plane(
        plane=np.sqrt(np.max(bead_img, axis=2)).T,
        spacing=spacing[1::-1],
        location=location[1::-1],
    )
    ax_yz.imshow(
        _num_to_nan(yz_plane),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        origin="lower",
    )

    _draw_fwhm(
        ax_yz,
        spacing=4000 / yz_plane.shape[1],
        fwhm=(fwhm_values[1], fwhm_values[0]),
        shape=yz_plane.shape,
    )

    base_ell, cv_ell = _add_ellipsoids(ax_3d, cov_matrix_3d, fwhm_values, spacing)

    _add_zyx_ellipsoid_projections(ax_3d, base_ell, cv_ell)

    principal_axes = np.sort(np.sqrt(np.linalg.eigvals(cov_matrix_3d)))[::-1]

    _add_principal_axes_text(
        ax_3d_text=ax_3d_text,
        date=date,
        fwhm_principal_axes=[fwhm(pa) for pa in principal_axes],
        version=version,
    )

    image = _fig_to_image(fig)

    plt.close(fig)
    return image


def _add_ellipsoids(
    ax_3d: Axes,
    cov_matrix_3d: ArrayLike,
    fwhm: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
) -> Tuple[Tuple[ArrayLike, ...], Tuple[ArrayLike, ...]]:
    base_ell = _get_ellipsoid(
        [
            [(fwhm[2] / (2 * np.sqrt(2 * np.log(2)))) ** 2, 0, 0],
            [0, (fwhm[1] / (2 * np.sqrt(2 * np.log(2)))) ** 2, 0],
            [0, 0, (fwhm[0] / (2 * np.sqrt(2 * np.log(2)))) ** 2],
        ],
        spacing,
    )
    ax_3d.plot_surface(
        *base_ell, rstride=2, cstride=2, color="white", antialiased=True, alpha=1
    )
    ax_3d.plot_wireframe(
        *base_ell, rstride=3, cstride=3, color="black", antialiased=True, alpha=0.5
    )
    cv_ell = _get_ellipsoid(cov_matrix_3d, spacing)
    ax_3d.plot_surface(
        *cv_ell, rstride=1, cstride=1, color="navy", antialiased=True, alpha=0.15
    )
    ax_3d.plot_wireframe(
        *cv_ell, rstride=4, cstride=4, color="navy", antialiased=True, alpha=0.25
    )
    ax_3d.set_box_aspect((1.0, 1.0, 4.0 / 3.0))
    ax_3d.set_xticks([-1000, 0, 1000])
    ax_3d.set_xticklabels(
        [-1000, 0, 1000], verticalalignment="bottom", horizontalalignment="right"
    )
    ax_3d.set_yticks([-1000, 0, 1000])
    ax_3d.set_yticklabels(
        [-1000, 0, 1000], verticalalignment="bottom", horizontalalignment="left"
    )
    ax_3d.set_zticks([-2000, -1000, 0, 1000, 2000])
    ax_3d.set_zticklabels(
        [-2000, -1000, 0, 1000, 2000],
        verticalalignment="top",
        horizontalalignment="left",
    )
    ax_3d.set_xlim(-1500, 1500)
    ax_3d.set_ylim(-1500, 1500)
    ax_3d.set_zlim(-2000, 2000)
    return base_ell, cv_ell


def _add_zyx_ellipsoid_projections(
    ax_3d: Axes,
    base_ell: Tuple[ArrayLike, ArrayLike, ArrayLike],
    cv_ell: Tuple[ArrayLike, ArrayLike, ArrayLike],
):
    ax_3d.contour(
        *base_ell,
        zdir="z",
        offset=ax_3d.get_zlim()[0],
        colors="white",
        levels=1,
        vmin=-1,
        vmax=1,
        zorder=0,
    )
    ax_3d.contour(
        *base_ell,
        zdir="y",
        offset=ax_3d.get_ylim()[1],
        colors="white",
        levels=1,
        vmin=-1,
        vmax=1,
        zorder=0,
    )
    ax_3d.contour(
        *base_ell,
        zdir="x",
        offset=ax_3d.get_xlim()[0],
        colors="white",
        levels=1,
        vmin=-1,
        vmax=1,
        zorder=0,
    )
    ax_3d.contour(
        *base_ell,
        zdir="z",
        offset=ax_3d.get_zlim()[0],
        colors="black",
        levels=1,
        vmin=-1,
        vmax=1,
        zorder=0,
        linestyles="--",
    )
    ax_3d.contour(
        *base_ell,
        zdir="y",
        offset=ax_3d.get_ylim()[1],
        colors="black",
        levels=1,
        vmin=-1,
        vmax=1,
        zorder=0,
        linestyles="--",
    )
    ax_3d.contour(
        *base_ell,
        zdir="x",
        offset=ax_3d.get_xlim()[0],
        colors="black",
        levels=1,
        vmin=-1,
        vmax=1,
        zorder=0,
        linestyles="--",
    )
    ax_3d.contour(
        *cv_ell,
        zdir="z",
        offset=ax_3d.get_zlim()[0],
        colors="navy",
        levels=1,
        vmin=-1,
        vmax=1,
        zorder=0,
    )
    ax_3d.contour(
        *cv_ell,
        zdir="y",
        offset=ax_3d.get_ylim()[1],
        colors="navy",
        levels=1,
        vmin=-1,
        vmax=1,
        zorder=0,
    )
    ax_3d.contour(
        *cv_ell,
        zdir="x",
        offset=ax_3d.get_xlim()[0],
        colors="navy",
        levels=1,
        vmin=-1,
        vmax=1,
        zorder=0,
    )


def _add_principal_axes_text(
    ax_3d_text: Axes,
    date: str,
    fwhm_principal_axes: Tuple[float, float, float],
    version: str,
):
    ax_3d_text.set_xlim(0, 100)
    ax_3d_text.set_ylim(0, 100)
    ax_3d_text.text(
        3,
        95,
        "Principal Axes",
        fontsize=14,
        weight="bold",
        color="navy",
    )
    ax_3d_text.text(
        5, 89, f"{int(np.round(fwhm_principal_axes[0]))}nm", fontsize=14, color="navy"
    )
    ax_3d_text.text(
        5, 83, f"{int(np.round(fwhm_principal_axes[1]))}nm", fontsize=14, color="navy"
    )
    ax_3d_text.text(
        5, 77, f"{int(np.round(fwhm_principal_axes[2]))}nm", fontsize=14, color="navy"
    )

    if date is not None:
        ax_3d_text.text(
            100,
            -4,
            f"Acquisition date: {date}",
            fontsize=11,
            horizontalalignment="right",
        )

    if version is not None:
        ax_3d_text.text(-110, -4, f"napari-psf-analysis: v{version}", fontsize=11)


def _fig_to_image(fig: Figure) -> ArrayLike:
    canvas = FigureCanvas(fig)
    canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
        (fig.canvas.get_width_height()[::-1]) + (3,)
    )
    return image


def _to_iso_plane(
    plane: ArrayLike, spacing: Tuple[float, float], location: Tuple[float, float]
) -> ArrayLike:
    plane_padded = np.pad(plane, 1, mode="constant", constant_values=(-1,))
    y, x = np.meshgrid(
        np.arange(plane_padded.shape[0]) * spacing[0] - location[0] - spacing[0],
        np.arange(plane_padded.shape[1]) * spacing[1] - location[1] - spacing[1],
        indexing="ij",
    )
    points = np.stack([y.ravel(), x.ravel()], axis=-1)

    num_samples = int(np.min(spacing) * 10)
    while num_samples < 1350:
        # 1350 pixels equals 300dpi for the configured figure size.
        num_samples = 2 * num_samples
    yy, xx = np.meshgrid(
        np.arange(num_samples) * 4000 / num_samples - 2000,
        np.arange(num_samples) * 4000 / num_samples - 2000,
        indexing="ij",
    )
    target_points = np.stack([yy.ravel(), xx.ravel()], axis=-1)
    return scipy.interpolate.griddata(
        points=points,
        values=plane_padded.ravel(),
        xi=target_points,
        method="nearest",
    ).reshape(num_samples, num_samples)


def _add_axes(fig: Figure) -> Tuple[Axes, Axes, Axes, Axes, Axes]:
    ax_xy = fig.add_axes([0.025, 0.525, 0.45, 0.45])
    ax_yz = fig.add_axes([0.525, 0.525, 0.45, 0.45])
    ax_zx = fig.add_axes([0.025, 0.025, 0.45, 0.45])
    ax_3d = fig.add_axes(
        [0.55, 0.025, 0.42, 0.42], projection="3d", computed_zorder=False
    )

    ax_X = fig.add_axes([0.25, 0.5, 0.02, 0.02])
    ax_X.text(0, -0.1, "X", fontsize=14, ha="center", va="center")
    ax_X.axis("off")
    ax_Y = fig.add_axes([0.5, 0.75, 0.02, 0.02])
    ax_Y.text(0, 0, "Y", fontsize=14, ha="center", va="center")
    ax_Y.axis("off")
    ax_Z = fig.add_axes([0.5, 0.25, 0.02, 0.02])
    ax_Z.text(-0.5, 0, "Z", fontsize=14, ha="center", va="center")
    ax_Z.axis("off")
    ax_Z1 = fig.add_axes([0.75, 0.5, 0.02, 0.02])
    ax_Z1.text(0, 0.5, "Z", fontsize=14, ha="center", va="center")
    ax_Z1.axis("off")
    ax_3d_text = fig.add_axes([0.525, 0.025, 0.45, 0.45])
    ax_3d_text.axis("off")

    return ax_xy, ax_yz, ax_zx, ax_3d, ax_3d_text


def localize_beads(
    img: ArrayLike,
    points: ArrayLike,
    patch_size: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
) -> Tuple[List[ArrayLike], List[Tuple[Any, ...]]]:
    """Crop sub-volumes around beads indicated by points.

    The extracted beads are centered in the sub-volumes.

    Parameters
    ----------
    img :
        Image containing beads
    points :
        Point coordinates indicating where a bead is located
    patch_size :
        Size of the sub-volume
    spacing :
        Voxel spacing of the image

    Returns
    -------
    beads
        A list of sub-volumes
    offsets
        The corresponding offsets relative to the image in voxel-coordinates
    """

    def _create_slice(mean, margin):
        start = int(mean - margin / 2)
        end = start + int(margin)
        return slice(start, end)

    beads = []
    offsets = []
    margins = np.array(patch_size) / np.array(spacing)
    for p in points:
        if np.all(p > margins / 2) and np.all(p < (np.array(img.shape) - margins / 2)):
            z_search_slice = _create_slice(p[0], margins[0])
            y_search_slice = _create_slice(p[1], margins[1])
            x_search_slice = _create_slice(p[2], margins[2])
            subvolume = img[z_search_slice, y_search_slice, x_search_slice]

            closest_roi = np.unravel_index(
                np.argmax(gaussian(subvolume, 2, mode="constant", preserve_range=True)),
                subvolume.shape,
            )
            offset = closest_roi - margins.astype(int) // 2
            z_slice = slice(
                offset[0] + z_search_slice.start, offset[0] + z_search_slice.stop
            )
            y_slice = slice(
                offset[1] + y_search_slice.start, offset[1] + y_search_slice.stop
            )
            x_slice = slice(
                offset[2] + x_search_slice.start, offset[2] + x_search_slice.stop
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


def merge(accumulated_results: dict, result: dict) -> dict:
    """Merge result dicts.

    Parameters
    ----------
    accumulated_results :
        All accumulated results
    result :
        New results

    Returns
    -------
    accumulated_results:
        Combination of both input dicts.
    """
    if accumulated_results is None:
        accumulated_results = {}
        for key in result.keys():
            accumulated_results[key] = [result[key]]
    else:
        for key in result.keys():
            accumulated_results[key].append(result[key])

    return accumulated_results


def create_result_table(results: dict) -> pd.DataFrame:
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
    return pd.DataFrame(
        {
            "ImageName": results["image_name"],
            "Date": results["date"],
            "Microscope": results["microscope"],
            "Magnification": results["mag"],
            "NA": results["NA"],
            "Amplitude_1D_Z": results["z_amp"],
            "Amplitude_2D_XY": results["yx_amp"],
            "Amplitude_3D_XYZ": results["zyx_amp"],
            "Background_1D_Z": results["z_bg"],
            "Background_2D_XY": results["yx_bg"],
            "Background_3D_XYZ": results["zyx_bg"],
            "Z_1D": results["z_mu"],
            "X_2D": results["x_mu"],
            "Y_2D": results["y_mu"],
            "X_3D": results["zyx_x_mu"],
            "Y_3D": results["zyx_y_mu"],
            "Z_3D": results["zyx_z_mu"],
            "FWHM_1D_Z": results["z_fwhm"],
            "FWHM_2D_X": results["x_fwhm"],
            "FWHM_2D_Y": results["y_fwhm"],
            "FWHM_3D_Z": results["zyx_z_fwhm"],
            "FWHM_3D_Y": results["zyx_y_fwhm"],
            "FWHM_3D_X": results["zyx_x_fwhm"],
            "FWHM_PA1_2D": results["yx_pc1_fwhm"],
            "FWHM_PA2_2D": results["yx_pc2_fwhm"],
            "FWHM_PA1_3D": results["zyx_pc1_fwhm"],
            "FWHM_PA2_3D": results["zyx_pc2_fwhm"],
            "FWHM_PA3_3D": results["zyx_pc3_fwhm"],
            "SignalToBG_1D_Z": (
                np.array(results["z_amp"]) / np.array(results["z_bg"])
            ).tolist(),
            "SignalToBG_2D_XY": (
                np.array(results["yx_amp"]) / np.array(results["yx_bg"])
            ).tolist(),
            "SignalToBG_3D_XYZ": (
                np.array(results["zyx_amp"]) / np.array(results["zyx_bg"])
            ).tolist(),
            "XYpixelsize": results["yx_spacing"],
            "Zspacing": results["z_spacing"],
            "cov_xx_3D": results["zyx_cxx"],
            "cov_xy_3D": results["zyx_cyx"],
            "cov_xz_3D": results["zyx_czx"],
            "cov_yy_3D": results["zyx_cyy"],
            "cov_yz_3D": results["zyx_czy"],
            "cov_zz_3D": results["zyx_czz"],
            "cov_xx_2D": results["yx_cxx"],
            "cov_xy_2D": results["yx_cyx"],
            "cov_yy_2D": results["yx_cyy"],
            "sde_amp_1D_Z": results["z_amp_sde"],
            "sde_amp_2D_XY": results["yx_amp_sde"],
            "sde_amp_3D_XYZ": results["zyx_amp_sde"],
            "sde_background_1D_Z": results["z_bg_sde"],
            "sde_background_2D_XY": results["yx_bg_sde"],
            "sde_background_3D_XYZ": results["zyx_bg_sde"],
            "sde_Z_1D": results["z_mu_sde"],
            "sde_X_2D": results["x_mu_sde"],
            "sde_Y_2D": results["y_mu_sde"],
            "sde_X_3D": results["zyx_x_mu_sde"],
            "sde_Y_3D": results["zyx_y_mu_sde"],
            "sde_Z_3D": results["zyx_z_mu_sde"],
            "sde_cov_xx_3D": results["zyx_cxx_sde"],
            "sde_cov_xy_3D": results["zyx_cyx_sde"],
            "sde_cov_xz_3D": results["zyx_czx_sde"],
            "sde_cov_yy_3D": results["zyx_cyy_sde"],
            "sde_cov_yz_3D": results["zyx_czx_sde"],
            "sde_cov_zz_3D": results["zyx_czz_sde"],
            "sde_cov_xx_2D": results["yx_cxx_sde"],
            "sde_cov_xy_2D": results["yx_cyx_sde"],
            "sde_cov_yy_2D": results["yx_cyy_sde"],
            "version": results["version"],
        }
    )
