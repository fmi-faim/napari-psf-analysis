from datetime import datetime

import matplotlib.axes
import numpy
import numpy as np
from matplotlib import pyplot as plt

from .gaussians import gaussian_1d
from .psf_lut import psf_lut


def add_gaussian_fits(
    ax_x: matplotlib.axes.Axes,
    ax_y: matplotlib.axes.Axes,
    mu_1: float,
    sigma_1: float,
    samples_1: list,
    mu_0: float,
    sigma_0: float,
    samples_0: list,
    height: float,
    offset: float,
    extent: tuple,
    scale: float = 4,
    aspect_ratio_x: float = 1,
    aspect_ratio_y: float = 1,
):
    """
    Add overlays with the measured data points and fitted Gaussians for X, Y,
    and Z dimensions.

    Parameters:
        ax_x: matplotlib.axes.Axes
            Figure x axis aligned plot.
        ax_y: matplotlib.axes.Axes
            Figure y axis aligned plot.
        mu_1: float
            Mean of the Gaussian plotted on ax_x.
        sigma_1: float
            Sigma of the Gaussian plotted on ax_x.
        samples_1: list[float]
            Measurements plotted on ax_x.
        mu_0: float
            Mean of the Gaussian plotted on ax_y.
        sigma_0: float
            Sigma of the Gaussian plotted on ax_y.
        samples_0: list[float]
            Measurements plotted on ax_y.
        height: float
            Height of the Gaussian.
        offset: float
            Offset of the Gaussian i.e. background signal value.
        extent: tuple(float, float)
            Extent of the fit range.
        scale: float
            Relative scaling of the plot.
        aspect_ratio_x: float
            Anisotropy for ax_x.
        aspect_ratio_y: float
            Anisotropy for ax_y.
    """
    fit_range = np.linspace(extent[0], extent[1], num=200)
    ax_x.plot(np.arange(len(samples_1)) + 0.5, samples_1, ".", color="black")
    ax_x.plot(
        fit_range + 0.5,
        gaussian_1d(height, mu_1, sigma_1, offset)(fit_range),
        color="gray",
    )
    ax_x.yaxis.set_ticks_position("left")
    ax_x.set_ylim([0, scale * height])
    ax_x.set_xlim([0, extent[1]])

    order = int(1)
    while height // order > 10:
        order = int(order * 10)

    hline_range = range(1 * order, int((height // order + 2) * order), order)
    ax_x.set_yticks(hline_range)
    ax_x.set_xticks([])
    ax_x.tick_params(axis="y", direction="in", pad=-35, grid_alpha=0.3, colors="gray")
    for yline in hline_range:
        ax_x.plot(
            [12 * aspect_ratio_x, extent[1]],
            [yline, yline],
            "gray",
            alpha=0.3,
            linewidth=1,
        )

    fit_range = np.linspace(0, extent[3] - extent[3] / (scale - 1), num=200)

    ax_y.plot(
        samples_0[: int(extent[3] - extent[3] / (scale - 1))],
        -1
        * (np.arange(len(samples_0)) + 0.5)[: int(extent[3] - extent[3] / (scale - 1))],
        ".",
        color="black",
    )
    ax_y.plot(
        gaussian_1d(height, mu_0, sigma_0, offset)(fit_range),
        -1 * (fit_range + 0.5),
        color="gray",
    )
    for yline in hline_range:
        ax_y.plot(
            [yline, yline],
            [-1 * (12 * aspect_ratio_y), -1 * (extent[3] - extent[3] / (scale - 1))],
            "gray",
            alpha=0.3,
            linewidth=1,
        )
    ax_y.set_ylim([-extent[3], 0])
    ax_y.set_xlim([0, scale * height])
    ax_y.set_xticks(hline_range)
    ax_y.set_yticks([])
    ax_y.tick_params(
        axis="x",
        direction="in",
        pad=-35,
        grid_alpha=0.3,
        labeltop=True,
        labelbottom=False,
        labelrotation=-90,
        top=True,
        bottom=False,
        colors="gray",
    )


def create_psf_overview(
    bead: numpy.array,
    params: tuple,
    vmin: float,
    vmax: float,
    extent: tuple,
    NA: float,
    fwhm_measures: list,
    cmap: matplotlib.colors.ListedColormap = psf_lut(),
):
    """
    Plots a single PSF with fitted Gaussian.

    Parameters:
        bead: numpy.array
            3D bead image.
        params: tuple(float)
            Parameters of fitted Gaussian: (height, mu_z, mu_y, mu_x, sigma_z,
            sigma_y, sigma_x, offset)
        vmin: float
            Display range minimum.
        vmax: float
            Display range maximum.
        extent: tuple
            Of the axes.
        NA: float
            Numerical aperture with which the bead was imaged.
        fwhm_measures: list[float]
            Full width half maxima measurements of the fitted Gaussian
            (z, y, x).
        cmap: matplotlib.colors.ListedColormap
            Colormap used to dispaly the bead image.

    Returns:
        fig: matplotlib.pyplot.figure
    """
    fig = plt.figure(figsize=(10, 10))

    ax_xy = fig.add_axes([0, 0.5, 0.45, 0.45])
    ax_yz = fig.add_axes([0.5, 0.5, 0.45, 0.45])
    ax_zx = fig.add_axes([0, 0, 0.45, 0.45])

    ax_xy.imshow(
        np.sqrt(np.max(bead, axis=0)),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        interpolation="nearest",
    )
    ax_xy.axis("off")
    ax_x = fig.add_axes([0, 0.5, 0.45, 0.45])
    ax_x.patch.set_alpha(0.0)
    ax_y = fig.add_axes([0, 0.5, 0.45, 0.45])
    ax_y.patch.set_alpha(0.0)
    add_gaussian_fits(
        ax_x,
        ax_y,
        params[3],
        params[6],
        bead[int(params[1]), int(params[2])],
        params[2],
        params[5],
        bead[int(params[1]), :, int(params[3])],
        params[0],
        params[-1],
        [0, bead.shape[2], 0, bead.shape[1]],
        scale=5,
    )
    ax_x.set_xlabel("X")
    ax_x.set_ylabel("Y")

    ax_zx.imshow(
        np.sqrt(np.max(bead, axis=1)),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        interpolation="nearest",
    )
    ax_zx.axis("off")
    ax_x = fig.add_axes([0, 0, 0.45, 0.45])
    ax_x.patch.set_alpha(0.0)
    ax_z = fig.add_axes([0, 0, 0.45, 0.45])
    ax_z.patch.set_alpha(0.0)
    add_gaussian_fits(
        ax_x,
        ax_z,
        params[3],
        params[6],
        bead[int(params[1]), int(params[2])],
        params[1],
        params[4],
        bead[:, int(params[2]), int(params[3])],
        params[0],
        params[-1],
        [0, bead.shape[2], 0, bead.shape[0]],
        scale=5,
        aspect_ratio_y=bead.shape[0] / bead.shape[2],
    )
    ax_x.set_xlabel("X")
    ax_x.set_ylabel("Z")

    ax_yz.imshow(
        np.sqrt(np.max(bead, axis=2)).T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        interpolation="nearest",
    )
    ax_yz.axis("off")
    ax_z = fig.add_axes([0.5, 0.5, 0.45, 0.45])
    ax_z.patch.set_alpha(0.0)
    ax_y = fig.add_axes([0.5, 0.5, 0.45, 0.45])
    ax_y.patch.set_alpha(0.0)
    add_gaussian_fits(
        ax_z,
        ax_y,
        params[1],
        params[4],
        bead[:, int(params[2]), int(params[3])],
        params[2],
        params[5],
        bead[int(params[1]), :, int(params[3])],
        params[0],
        params[-1],
        [0, bead.shape[0], 0, bead.shape[1]],
        scale=5,
        aspect_ratio_x=bead.shape[0] / bead.shape[1],
    )
    ax_z.set_xlabel("Z")
    ax_z.set_ylabel("Y")

    ax_text = fig.add_axes([0.5, 0, 0.45, 0.45])
    ax_text.axis("off")
    red_font = {"color": "red", "size": 15}
    ax_text.text(0.0, 0.40, datetime.today().strftime("%d-%m-%Y"), fontdict=red_font)
    ax_text.text(
        0.0,
        0.34,
        "FWHM lateral: X = {:.0f}nm, Y = {:.0f}nm".format(
            fwhm_measures[2], fwhm_measures[1]
        ),
        fontdict=red_font,
    )
    ax_text.text(
        0.0,
        0.28,
        "FWHM lateral average = {:.0f}nm".format(
            (fwhm_measures[2] + fwhm_measures[1]) / 2
        ),
        fontdict=red_font,
    )
    ax_text.text(0.0, 0.22, f"FWHM axial = {fwhm_measures[0]:.0f}nm", fontdict=red_font)
    blue_font = {"color": "blue", "size": 15}
    ax_text.text(0.0, 0.16, "Theoretical values", fontdict=blue_font)
    if NA == 1.4:
        ax_text.text(
            0.0, 0.1, "NA1.4 Oil FWHMl 188nm - FWHMa 696nm", fontdict=blue_font
        )
    elif NA == 1.32:
        ax_text.text(
            0.0, 0.1, "NA1.32 Oil FWHMl 199nm - FWHMa 781nm", fontdict=blue_font
        )
    elif NA == 1.3:
        ax_text.text(
            0.0, 0.1, "NA1.3 Oil FWHMl 202nm - FWHMa 803nm", fontdict=blue_font
        )
        ax_text.text(
            0.0, 0.04, "NA1.3 Glyc FWHMl 202nm - FWHMa 771nm", fontdict=blue_font
        )
    elif NA == 0.95:
        ax_text.text(
            0.0, 0.1, "NA0.95 Air FWHMl 276nm - FWHMa 996nm", fontdict=blue_font
        )
    elif NA == 1.2:
        ax_text.text(
            0.0, 0.1, "NA1.2 Water FWHMl 213nm - FWHMa 799nm", fontdict=blue_font
        )
    elif NA == 0.8:
        ax_text.text(
            0.0, 0.1, "NA0.8 Glyc FWHMl 328nm - FWHMa 2038nm", fontdict=blue_font
        )
        ax_text.text(
            0.0, 0.04, "NA0.8 Air FWHMl 383nm - FWHMa 1403nm", fontdict=blue_font
        )
    elif NA == 0.75:
        ax_text.text(
            0.0, 0.1, "NA0.75 Air FWHMl 350nm - FWHMa 1598nm", fontdict=blue_font
        )
    elif NA == 1.45:
        ax_text.text(
            0.0, 0.1, "NA1.45 Oil FWHMl 181nm - FWHMa 649nm", fontdict=blue_font
        )
    elif NA == 1.25:
        ax_text.text(
            0.0, 0.1, "NA1.25 Oil FWHMl 210nm - FWHMa 873nm", fontdict=blue_font
        )
    else:
        ax_text.text(0.0, 0.1, "Values not determined yet.", fontdict=blue_font)

    return fig
