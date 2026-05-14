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


PNG_OUT = OUT_DIR / "figure_01_direction_disk.png"
PDF_OUT = OUT_DIR / "figure_01_direction_disk.pdf"


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
    disk_dir = unit(disk_angle)
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

    disk_axis = 3.4 * disk_dir
    radius_end = -disk_axis
    draw_line(ax, origin, radius_end, color="black", lw=1.7, zorder=5)

    arrow_start = origin
    arrow_end = 1.8 * nu_dir
    ax.add_patch(
        FancyArrowPatch(
            arrow_start,
            arrow_end,
            arrowstyle="-|>",
            mutation_scale=18,
            lw=2.1,
            color="black",
            zorder=6,
        )
    )

    # Right-angle marker: the sampled disk is perpendicular to the neutrino path.
    marker_size = 0.28
    radius_dir = radius_end / np.linalg.norm(radius_end)
    right_angle_points = np.array(
        [
            marker_size * nu_dir,
            marker_size * nu_dir + marker_size * radius_dir,
            marker_size * radius_dir,
        ]
    )
    ax.plot(
        right_angle_points[:, 0],
        right_angle_points[:, 1],
        color="black",
        lw=1.6,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=9,
    )

    ax.scatter([0], [0], s=48, color="black", zorder=8)

    ax.text(
        -0.5,
        -0.12,
        "detector origin",
        ha="left",
        va="top",
        color="#222222",
        fontsize=10,
    )
    ax.text(
        *(arrow_end + np.array([-0.8, 0.03])),
        "sampled neutrino direction",
        ha="left",
        va="bottom",
        color="#111111",
        fontsize=10,
    )
    ax.text(
        *(arrow_end + np.array([0.08, -0.02])),
        "",
        ha="left",
        va="center",
        color="#111111",
        fontsize=9.5,
    )
    ax.text(
        *(0.55 * radius_end + np.array([-0.12, 0.18])),
        r"$R_{\mathrm{inj}}$",
        ha="center",
        va="bottom",
        color="#111111",
        fontsize=11,
    )
    ax.set_xlim(-4.6, 4.8)
    ax.set_ylim(-3.3, 3.35)
    fig.tight_layout(pad=0.25)

    # fig.savefig(PNG_OUT, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(PDF_OUT, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(PNG_OUT)
    print(PDF_OUT)


if __name__ == "__main__":
    main()
