import numpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib_inline.backend_inline import FigureCanvas


def get_parameters(popt):
    A = popt[0]
    B = popt[1]
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
    return A, B, mu_x, mu_y, mu_z, cov


def create_psf_overview(
    bead: numpy.array,
    params: tuple,
    vmin: float,
    vmax: float,
    extent: tuple,
    principal_axes: list,
    fwhm: list,
    fwhm_2d: list,
    bbox_size: int,
    xy_spacing: int,
    z_spacing: int,
    date: str,
    version: str,
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
        principal_axes: list[float]
            Full width half maxima measurements of the fitted Gaussian
            (pa1, pa2, pa3).
        fwhm: list[float]
            Full width half maxima measurements of the fitted Gaussian
            (x, y, z).
        fwhm_2d: list[float]
            Full width half maxima measurements of the fitted Gaussian
            (x, y).
        bbox_size: int
            Size of the bounding box.
        xy_spacing: int
            Pixel spacing in xy.
        z_spacing: int
            Z-slice spacing.
        date: str
            Date as string.
        version: str
            Plugin version

    Returns:
        fig: matplotlib.pyplot.figure
    """
    height, background, mu_x, mu_y, mu_z, estimated_cov = get_parameters(params)
    eigval, eigvec = np.linalg.eig(estimated_cov)
    fig = plt.figure(figsize=(10, 10))

    ax_xy = fig.add_axes([0.025, 0.525, 0.45, 0.45])
    ax_yz = fig.add_axes([0.525, 0.525, 0.45, 0.45])
    ax_zx = fig.add_axes([0.025, 0.025, 0.45, 0.45])

    ax_X = fig.add_axes([0.24, 0.49, 0.02, 0.02])
    ax_X.text(0.5, 0.5, "X", fontsize=14, ha="center", va="center")
    ax_X.axis("off")
    ax_Y = fig.add_axes([0.49, 0.74, 0.02, 0.02])
    ax_Y.text(0.5, 0.5, "Y", fontsize=14, ha="center", va="center")
    ax_Y.axis("off")
    ax_Z = fig.add_axes([0.477, 0.24, 0.02, 0.02])
    ax_Z.text(0.5, 0.5, "Z", fontsize=14, ha="center", va="center")
    ax_Z.axis("off")
    ax_Z1 = fig.add_axes([0.73, 0.50, 0.02, 0.02])
    ax_Z1.text(0.5, 0.5, "Z", fontsize=14, ha="center", va="center")
    ax_Z1.axis("off")

    cx = mu_x / xy_spacing + 0.5
    cy = mu_y / xy_spacing + 0.5
    cz = (mu_z + z_spacing * 0.5) / xy_spacing + 0.5

    ax_xy.imshow(
        np.sqrt(np.max(bead, axis=0)),
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        interpolation="nearest",
        origin="lower",
    )
    x_fwhm, y_fwhm, z_fwhm = fwhm
    x_fwhm_2d, y_fwhm_2d = fwhm_2d
    dx = (x_fwhm / 2) / xy_spacing
    dy = (y_fwhm / 2) / xy_spacing
    ax_xy.plot(
        [cx - dx, cx + dx],
        [
            bead.shape[2] / 4,
        ]
        * 2,
        linewidth=4,
        c="white",
        solid_capstyle="butt",
    )
    ax_xy.plot([cx - dx, cx - dx], [cy - dx, bead.shape[2] / 4], "--", c="white")
    ax_xy.plot([cx + dx, cx + dx], [cy - dx, bead.shape[2] / 4], "--", c="white")
    ax_xy.text(
        cx,
        bead.shape[2] / 4.5,
        f"{int(np.round(x_fwhm))}nm",
        ha="center",
        va="top",
        fontsize=15,
        bbox=dict(facecolor="white", alpha=1, linewidth=0),
    )

    ax_xy.plot(
        [
            bead.shape[1] / 4,
        ]
        * 2,
        [cy - dy, cy + dy],
        linewidth=4,
        c="white",
        solid_capstyle="butt",
    )
    ax_xy.plot([bead.shape[1] / 4, cx - dx], [cy - dy, cy - dy], "--", color="white")
    ax_xy.plot([bead.shape[1] / 4, cx - dx], [cy + dy, cy + dy], "--", color="white")
    ax_xy.text(
        bead.shape[2] / 4.5,
        cy,
        f"{int(np.round(y_fwhm))}nm",
        ha="right",
        va="center",
        fontsize=15,
        bbox=dict(facecolor="white", alpha=1, linewidth=0),
    )
    ax_xy.text(
        0,
        0,
        f"2D FWHM Fit [nm]:\n"
        f"(x, y) = ({int(np.round(x_fwhm_2d))}, {int(np.round(y_fwhm_2d))})",
        ha="left",
        va="bottom",
        fontsize=15,
        bbox=dict(facecolor="white", alpha=1, linewidth=0),
    )

    from matplotlib_scalebar.scalebar import ScaleBar

    scalebar = ScaleBar(xy_spacing, "nm", fixed_value=500, location="lower " "right")
    ax_xy.add_artist(scalebar)
    ax_xy.set_xticks([])
    ax_xy.set_yticks([])

    ax_zx.imshow(
        np.sqrt(np.max(bead, axis=1)),
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        interpolation="nearest",
        origin="lower",
    )
    dz = (z_fwhm / 2) / xy_spacing
    ax_zx.plot(
        [
            bead.shape[1] / 4,
        ]
        * 2,
        [cz - dz, cz + dz],
        linewidth=4,
        c="white",
        solid_capstyle="butt",
    )
    ax_zx.plot([bead.shape[1] / 4, cx - dx], [cz - dz, cz - dz], "--", color="white")
    ax_zx.plot([bead.shape[1] / 4, cx - dx], [cz + dz, cz + dz], "--", color="white")
    ax_zx.text(
        bead.shape[1] / 4.5,
        cz,
        f"{int(np.round(z_fwhm))}nm",
        ha="right",
        va="center",
        fontsize=15,
        bbox=dict(facecolor="white", alpha=1, linewidth=0),
    )
    ax_zx.plot(
        [cx - dx, cx + dx],
        [
            bead.shape[2] / 4,
        ]
        * 2,
        linewidth=4,
        c="white",
        solid_capstyle="butt",
    )
    ax_zx.plot([cx - dx, cx - dx], [bead.shape[1] / 4, cz - dz], "--", color="white")
    ax_zx.plot([cx + dx, cx + dx], [bead.shape[1] / 4, cz - dz], "--", color="white")
    ax_zx.text(
        cx,
        bead.shape[2] / 4.5,
        f"{int(np.round(x_fwhm))}nm",
        ha="center",
        va="top",
        fontsize=15,
        bbox=dict(facecolor="white", alpha=1, linewidth=0),
    )

    ax_zx.set_xticks([])
    ax_zx.set_yticks([])

    ax_yz.imshow(
        np.sqrt(np.max(bead, axis=2)).T,
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        interpolation="nearest",
        origin="lower",
    )
    ax_yz.plot(
        [
            bead.shape[1] / 4,
        ]
        * 2,
        [cy - dy, cy + dy],
        linewidth=4,
        c="white",
        solid_capstyle="butt",
    )
    ax_yz.plot([bead.shape[1] / 4, cz - dz], [cy - dy, cy - dy], "--", color="white")
    ax_yz.plot([bead.shape[1] / 4, cz - dz], [cy + dy, cy + dy], "--", color="white")
    ax_yz.text(
        bead.shape[2] / 4.5,
        cy,
        f"{int(np.round(y_fwhm))}nm",
        ha="right",
        va="center",
        fontsize=15,
        bbox=dict(facecolor="white", alpha=1, linewidth=0),
    )
    ax_yz.plot(
        [cz - dz, cz + dz],
        [
            bead.shape[1] / 4,
        ]
        * 2,
        linewidth=4,
        c="white",
        solid_capstyle="butt",
    )
    ax_yz.plot([cz - dz, cz - dz], [bead.shape[1] / 4, cy - dy], "--", color="white")
    ax_yz.plot([cz + dz, cz + dz], [bead.shape[1] / 4, cy - dy], "--", color="white")
    ax_yz.text(
        cz,
        bead.shape[1] / 4.5,
        f"{int(np.round(z_fwhm))}nm",
        ha="center",
        va="top",
        fontsize=15,
        bbox=dict(facecolor="white", alpha=1, linewidth=0),
    )

    ax_yz.set_xticks([])
    ax_yz.set_yticks([])

    ax_3D = fig.add_axes(
        [0.525, 0.025, 0.45, 0.45], projection="3d", computed_zorder=False
    )
    ax_text = fig.add_axes([0.525, 0.025, 0.45, 0.45])
    ax_3D.axis("off")
    ax_3D.set_xlim(-bbox_size, bbox_size)
    ax_3D.set_ylim(-bbox_size, bbox_size)
    ax_3D.set_zlim(-bbox_size, bbox_size)
    ax_3D.plot(
        [0.5 * bbox_size, 0.5 * bbox_size],
        [-0.5 * bbox_size, 0.5 * bbox_size],
        [-bbox_size, -bbox_size],
        c="#444444",
        linewidth=1,
    )
    ax_3D.plot(
        [0.5 * bbox_size, -0.5 * bbox_size],
        [0.5 * bbox_size, 0.5 * bbox_size],
        [-bbox_size, -bbox_size],
        c="#444444",
        linewidth=1,
    )
    ax_3D.plot(
        [0.5 * bbox_size, 0.5 * bbox_size],
        [0.5 * bbox_size, 0.5 * bbox_size],
        [-bbox_size, bbox_size],
        c="#444444",
        linewidth=1,
    )
    ax_3D.plot(
        [-0.5 * bbox_size, -0.5 * bbox_size],
        [0.5 * bbox_size, -0.5 * bbox_size],
        [bbox_size, bbox_size],
        c="#444444",
        linewidth=1,
    )
    ax_3D.plot(
        [0.5 * bbox_size, -0.5 * bbox_size],
        [0.5 * bbox_size, 0.5 * bbox_size],
        [bbox_size, bbox_size],
        c="#444444",
        linewidth=1,
    )
    ax_3D.plot(
        [-0.5 * bbox_size, -0.5 * bbox_size],
        [0.5 * bbox_size, -0.5 * bbox_size],
        [-bbox_size, -bbox_size],
        c="#444444",
        linewidth=1,
    )
    ax_3D.plot(
        [0.5 * bbox_size, -0.5 * bbox_size],
        [-0.5 * bbox_size, -0.5 * bbox_size],
        [-bbox_size, -bbox_size],
        c="#444444",
        linewidth=1,
    )
    ax_3D.plot(
        [-0.5 * bbox_size, -0.5 * bbox_size],
        [0.5 * bbox_size, 0.5 * bbox_size],
        [-bbox_size, bbox_size],
        c="#444444",
        linewidth=1,
    )
    ax_3D.plot(
        [-0.5 * bbox_size, -0.5 * bbox_size],
        [-0.5 * bbox_size, -0.5 * bbox_size],
        [-bbox_size, bbox_size],
        c="#444444",
        linewidth=1,
    )
    ax_3D.plot(
        [-0.5 * bbox_size, -0.5 * bbox_size],
        [0.5 * bbox_size, -0.5 * bbox_size],
        [0, 0],
        c="#444444",
        linewidth=1,
    )
    ax_3D.plot(
        [-0.5 * bbox_size, -0.5 * bbox_size],
        [0, 0],
        [-bbox_size, bbox_size],
        c="#444444",
        linewidth=1,
    )
    ax_3D.plot(
        [0.5 * bbox_size, -0.5 * bbox_size],
        [0.5 * bbox_size, 0.5 * bbox_size],
        [0, 0],
        c="#444444",
        linewidth=1,
    )
    ax_3D.plot(
        [0, 0],
        [0.5 * bbox_size, 0.5 * bbox_size],
        [-bbox_size, bbox_size],
        c="#444444",
        linewidth=1,
    )
    ax_3D.plot(
        [0.5 * bbox_size, -0.5 * bbox_size],
        [0, 0],
        [-bbox_size, -bbox_size],
        c="#444444",
        linewidth=1,
    )
    ax_3D.plot(
        [0, 0],
        [-0.5 * bbox_size, 0.5 * bbox_size],
        [-bbox_size, -bbox_size],
        c="#444444",
        linewidth=1,
    )

    ax_3D.plot([-0.5 * bbox_size, 0], [0, 0], [0, 0], "--", c="#444444", linewidth=1)
    ax_3D.plot([0, 0], [0, 0.5 * bbox_size], [0, 0], "--", c="#444444", linewidth=1)
    ax_3D.plot([0, 0], [0, 0], [0, -bbox_size], "--", c="#444444", linewidth=1)

    neg = -eigvec[:, 0] * principal_axes[0] / 2.0
    pos = eigvec[:, 0] * principal_axes[0] / 2.0
    ax_3D.plot(
        [neg[0], pos[0]],
        [0.5 * bbox_size, 0.5 * bbox_size],
        [neg[2], pos[2]],
        color="#0061B5",
        linestyle=(0, (1, 0.5)),
        linewidth=3,
    )
    ax_3D.plot(
        [-0.5 * bbox_size, -0.5 * bbox_size],
        [neg[1], pos[1]],
        [neg[2], pos[2]],
        color="#0061B5",
        linestyle=(0, (1, 0.5)),
        linewidth=3,
    )
    ax_3D.plot(
        [neg[0], pos[0]],
        [neg[1], pos[1]],
        [-bbox_size, -bbox_size],
        color="#0061B5",
        linestyle=(0, (1, 0.5)),
        linewidth=3,
    )
    neg = -eigvec[:, 1] * principal_axes[1] / 2.0
    pos = eigvec[:, 1] * principal_axes[1] / 2.0
    ax_3D.plot(
        [neg[0], pos[0]],
        [0.5 * bbox_size, 0.5 * bbox_size],
        [neg[2], pos[2]],
        color="#D81B60",
        linestyle=(0, (1, 0.5)),
        linewidth=3,
    )
    ax_3D.plot(
        [-0.5 * bbox_size, -0.5 * bbox_size],
        [neg[1], pos[1]],
        [neg[2], pos[2]],
        color="#D81B60",
        linestyle=(0, (1, 0.5)),
        linewidth=3,
    )
    ax_3D.plot(
        [neg[0], pos[0]],
        [neg[1], pos[1]],
        [-bbox_size, -bbox_size],
        color="#D81B60",
        linestyle=(0, (1, 0.5)),
        linewidth=3,
    )
    neg = -eigvec[:, 2] * principal_axes[2] / 2.0
    pos = eigvec[:, 2] * principal_axes[2] / 2.0
    ax_3D.plot(
        [neg[0], pos[0]],
        [0.5 * bbox_size, 0.5 * bbox_size],
        [neg[2], pos[2]],
        color="#03A919",
        linestyle=(0, (1, 0.5)),
        linewidth=3,
    )
    ax_3D.plot(
        [-0.5 * bbox_size, -0.5 * bbox_size],
        [neg[1], pos[1]],
        [neg[2], pos[2]],
        color="#03A919",
        linestyle=(0, (1, 0.5)),
        linewidth=3,
    )
    ax_3D.plot(
        [neg[0], pos[0]],
        [neg[1], pos[1]],
        [-bbox_size, -bbox_size],
        color="#03A919",
        linestyle=(0, (1, 0.5)),
        linewidth=3,
    )

    ax_3D.text3D(-0.5 * bbox_size, -0.64 * bbox_size, 0, "Z", ha="center", va="center")
    ax_3D.text3D(0, -0.6 * bbox_size, -1.05 * bbox_size, "X", ha="center", va="center")
    ax_3D.text3D(0.6 * bbox_size, 0, -bbox_size, "Y", ha="center", va="center")

    neg = -eigvec[:, 0] * principal_axes[0] / 2.0
    pos = eigvec[:, 0] * principal_axes[0] / 2.0
    ax_3D.quiver3D(
        0, 0, 0, *neg, linewidth=3, zorder=2, arrow_length_ratio=0, color="#0061B5"
    )
    ax_3D.quiver3D(
        0, 0, 0, *pos, linewidth=3, zorder=2, arrow_length_ratio=0, color="#0061B5"
    )
    neg = -eigvec[:, 1] * principal_axes[1] / 2.0
    pos = eigvec[:, 1] * principal_axes[1] / 2.0
    ax_3D.quiver3D(
        0, 0, 0, *neg, linewidth=3, zorder=2, arrow_length_ratio=0, color="#D81B60"
    )
    ax_3D.quiver3D(
        0, 0, 0, *pos, linewidth=3, zorder=2, arrow_length_ratio=0, color="#D81B60"
    )
    neg = -eigvec[:, 2] * principal_axes[2] / 2.0
    pos = eigvec[:, 2] * principal_axes[2] / 2.0
    ax_3D.quiver3D(
        0, 0, 0, *neg, linewidth=3, zorder=2, arrow_length_ratio=0, color="#03A919"
    )
    ax_3D.quiver3D(
        0, 0, 0, *pos, linewidth=3, zorder=2, arrow_length_ratio=0, color="#03A919"
    )

    ax_text.axis("off")
    ax_text.set_xlim(0, 100)
    ax_text.set_ylim(0, 100)
    ax_text.text(
        5,
        90,
        f"{int(np.round(principal_axes[0]))}nm",
        color="#0061B5",
        fontsize=16,
        weight="bold",
    )
    ax_text.text(
        5,
        83,
        f"{int(np.round(principal_axes[1]))}nm",
        color="#D81B60",
        fontsize=16,
        weight="bold",
    )
    ax_text.text(
        5,
        76,
        f"{int(np.round(principal_axes[2]))}nm",
        color="#03A919",
        fontsize=16,
        weight="bold",
    )
    ax_text.text(25, 2, f"Acquisition date: {date}.", fontsize=16)
    ax_text.text(-110, -4, f"napari-psf-analysis: v{version}", fontsize=11)

    ax_text.plot([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], color="black")

    canvas = FigureCanvas(fig)
    canvas.draw()  # draw the canvas, cache the renderer

    image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
        (fig.canvas.get_width_height()[::-1]) + (3,)
    )

    plt.close(fig)
    return image
