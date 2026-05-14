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


PNG_OUT = OUT_DIR / "figure_02_pca_point.png"
PDF_OUT = OUT_DIR / "figure_02_pca_point.pdf"


def unit(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.array([np.cos(angle), np.sin(angle)])


def draw_line(ax, p1, p2, **kwargs):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)


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
    axis_style = dict(color="#b2b2b2", lw=3.8, solid_capstyle="round", zorder=1)
    draw_line(ax, np.array([-4.1, 0.0]), np.array([4.3, 0.0]), **axis_style)
    draw_line(ax, np.array([0.0, -3.0]), np.array([0.0, 3.0]), **axis_style)
    draw_line(
        ax,
        np.array([-3.75, -0.94]),
        np.array([3.9, 0.98]),
        color="#c4c4c4",
        lw=3.8,
        solid_capstyle="round",
        zorder=0,
    )

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
            lw=2.0,
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
            lw=2.3,
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
        lw=1.2,
        linestyle=(0, (4, 3)),
        zorder=6,
    )

    arrow_start = pca
    arrow_end = pca + 1.05 * nu_dir
    draw_line(
        ax,
        pca - 2.8 * nu_dir,
        pca,
        color="black",
        lw=1.35,
        linestyle=(0, (4, 3)),
        alpha=0.72,
        zorder=1.7,
    )
    draw_line(
        ax,
        arrow_end,
        pca + 5 * nu_dir,
        color="black",
        lw=1.35,
        linestyle=(0, (4, 3)),
        zorder=8,
    )
    ax.add_patch(
        FancyArrowPatch(
            arrow_start,
            arrow_end,
            arrowstyle="-|>",
            mutation_scale=15,
            lw=1.9,
            color="black",
            zorder=9,
        )
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
        lw=1.4,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=10,
    )

    ax.scatter(
        [pca[0]],
        [pca[1]],
        s=82,
        color="#c9252d",
        edgecolor="#7a1016",
        linewidth=0.8,
        zorder=11,
    )
    ax.text(
        *(pca + np.array([0.1, -0.28])),
        "PCA",
        ha="left",
        va="bottom",
        color="#c9252d",
        fontsize=11,
        fontweight="bold",
        zorder=12,
    )

    ax.scatter([0], [0], s=48, color="black", zorder=8)
    ax.set_xlim(-4.6, 4.8)
    ax.set_ylim(-3.3, 3.35)
    fig.tight_layout(pad=0.25)

    # fig.savefig(PNG_OUT, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(PDF_OUT, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    # print(PNG_OUT)
    print(PDF_OUT)


if __name__ == "__main__":
    main()
