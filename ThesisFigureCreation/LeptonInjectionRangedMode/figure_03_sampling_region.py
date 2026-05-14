from pathlib import Path
import os

import numpy as np


OUT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUT_DIR / ".cache"))
os.environ.setdefault("FC_CACHEDIR", str(OUT_DIR / ".cache" / "fontconfig"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch


mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


PNG_OUT = OUT_DIR / "figure_03_sampling_region.png"
PDF_OUT = OUT_DIR / "figure_03_sampling_region.pdf"

VIEW_XLIM = (-10.4, 10.4)
VIEW_YLIM = (-10.2, 8.0)
REFERENCE_VIEW_WIDTH = 9.4
STYLE_SCALE = min(1.0, REFERENCE_VIEW_WIDTH / (VIEW_XLIM[1] - VIEW_XLIM[0]))


def lw(value: float) -> float:
    return value * STYLE_SCALE


def area(value: float) -> float:
    return value * STYLE_SCALE**2


def font(value: float) -> float:
    return max(5.5, value * STYLE_SCALE)


def unit(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.array([np.cos(angle), np.sin(angle)])


def draw_line(ax, p1, p2, **kwargs):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)


def draw_tick(ax, point, direction, length=0.18, **kwargs):
    normal = np.array([-direction[1], direction[0]])
    draw_line(ax, point - 0.5 * length * normal, point + 0.5 * length * normal, **kwargs)


def draw_axis_through_origin(ax, direction, negative_scale=1.0, positive_scale=1.0, **kwargs):
    view_span = 0.42 * max(VIEW_XLIM[1] - VIEW_XLIM[0], VIEW_YLIM[1] - VIEW_YLIM[0])
    draw_line(ax, -negative_scale * view_span * direction, positive_scale * view_span * direction, **kwargs)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")

    origin = np.array([0.0, 0.0])

    direction_angle = 52
    disk_angle = direction_angle - 90
    nu_dir = unit(direction_angle)

    # Detector-coordinate axes, kept subtle so the injection construction dominates.
    axis_style = dict(color="#b2b2b2", lw=lw(3.8), solid_capstyle="round", zorder=1)
    draw_axis_through_origin(ax, np.array([1.0, 0.0]), **axis_style)
    draw_axis_through_origin(ax, np.array([0.0, 1.0]), negative_scale=1.0, positive_scale=0.66, **axis_style)
    draw_axis_through_origin(ax, unit(14), color="#c4c4c4", lw=lw(3.8), solid_capstyle="round", zorder=0)

    # Disk appears as an ellipse because the disk plane is viewed obliquely.
    # A small offset copy and split outline give a simple pseudo-3D thickness.
    disk_shadow = Ellipse(
        xy=origin + np.array([0.08, -0.08]),
        width=6.8,
        height=1.45,
        angle=disk_angle,
        facecolor="#bfbfbf",
        edgecolor="none",
        alpha=0.18,
        zorder=1,
    )
    ax.add_patch(disk_shadow)

    disk = Ellipse(
        xy=origin,
        width=6.8,
        height=1.45,
        angle=disk_angle,
        facecolor="#d9d9d9",
        edgecolor="none",
        alpha=0.58,
        zorder=2,
    )
    ax.add_patch(disk)

    ax.add_patch(
        Arc(
            origin,
            width=6.8,
            height=1.45,
            angle=disk_angle,
            theta1=0,
            theta2=180,
            color="#8a8a8a",
            lw=lw(2.0),
            zorder=3,
        )
    )
    ax.add_patch(
        Arc(
            origin,
            width=6.8,
            height=1.45,
            angle=disk_angle,
            theta1=180,
            theta2=360,
            color="#303030",
            lw=lw(2.3),
            zorder=4,
        )
    )

    # The sampled disk point is the point of closest approach (PCA).
    a_ell = 6.8 / 2
    b_ell = 1.45 / 2
    pca_t = 0
    pca_r = 0.50
    theta = np.deg2rad(disk_angle)
    pca = pca_r * np.array(
        [
            a_ell * np.cos(pca_t) * np.cos(theta)
            - b_ell * np.sin(pca_t) * np.sin(theta),
            a_ell * np.cos(pca_t) * np.sin(theta)
            + b_ell * np.sin(pca_t) * np.cos(theta),
        ]
    )
    draw_line(
        ax,
        origin,
        pca,
        color="black",
        lw=lw(1.2),
        linestyle=(0, (4, 3)),
        zorder=6,
    )

    endcap_length = 2.55
    range_length = 6.2
    endcap_start = pca - endcap_length * nu_dir
    endcap_stop = pca + endcap_length * nu_dir
    range_start = endcap_start - range_length * nu_dir

    arrow_start = pca
    arrow_end = pca + 0.9 * nu_dir
    draw_line(
        ax,
        range_start - 1.7 * nu_dir,
        endcap_stop + 3.15 * nu_dir,
        color="black",
        lw=lw(1.15),
        linestyle=(0, (4, 3)),
        alpha=0.75,
        zorder=7,
    )
    ax.add_patch(
        FancyArrowPatch(
            arrow_start,
            arrow_end,
            arrowstyle="-|>",
            mutation_scale=15 * STYLE_SCALE,
            lw=lw(1.9),
            color="black",
            zorder=11,
        )
    )

    blue = "#1f77b4"
    range_color = "#2ca25f"
    pca_red = "#b2182b"
    pca_edge = "#7f1d2d"
    draw_line(
        ax,
        endcap_start,
        endcap_stop,
        color=blue,
        lw=lw(2.2),
        solid_capstyle="round",
        zorder=9,
    )
    draw_tick(ax, endcap_start, nu_dir, length=0.20, color=blue, lw=lw(1.6), zorder=10)
    draw_tick(ax, endcap_stop, nu_dir, length=0.20, color=blue, lw=lw(1.6), zorder=10)
    ax.text(
        *(pca + 0.62 * endcap_length * nu_dir + np.array([-0.15, -0.70])),
        r"$2\times$ EndcapLength",
        ha="left",
        va="center",
        color=blue,
        fontsize=font(12),
        zorder=12,
    )

    draw_line(
        ax,
        range_start,
        endcap_start,
        color=range_color,
        lw=lw(2.1),
        solid_capstyle="round",
        zorder=9,
    )
    draw_tick(ax, range_start, nu_dir, length=0.22, color=range_color, lw=lw(1.6), zorder=10)
    ax.text(
        *(endcap_start - 0.48 * range_length * nu_dir + np.array([0.15, -0.12])),
        "Range",
        ha="left",
        va="center",
        color=range_color,
        fontsize=font(15),
        zorder=12,
    )

    marker_size = 0.22
    pca_to_origin_dir = -pca / np.linalg.norm(pca)
    right_angle_points = np.array(
        [
            pca + marker_size * nu_dir,
            pca + marker_size * nu_dir + marker_size * pca_to_origin_dir,
            pca + marker_size * pca_to_origin_dir,
        ]
    )
    ax.plot(
        right_angle_points[:, 0],
        right_angle_points[:, 1],
        color="black",
        lw=lw(1.4),
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=10,
    )

    ax.scatter(
        [pca[0]],
        [pca[1]],
        s=area(82),
        color=pca_red,
        edgecolor=pca_edge,
        linewidth=lw(0.8),
        zorder=11,
    )
    ax.text(
        *(pca + np.array([0.1, -0.28])),
        "PCA",
        ha="left",
        va="bottom",
        color=pca_red,
        fontsize=font(11),
        fontweight="bold",
        zorder=12,
    )

    ax.scatter([0], [0], s=area(48), color="black", zorder=8)

    ax.set_xlim(*VIEW_XLIM)
    ax.set_ylim(*VIEW_YLIM)
    fig.tight_layout(pad=0.25)

    # fig.savefig(PNG_OUT, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(PDF_OUT, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    # print(PNG_OUT)
    print(PDF_OUT)


if __name__ == "__main__":
    main()
