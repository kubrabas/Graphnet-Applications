import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon

# -----------------------------
# Figure setup
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# -----------------------------
# Helper
# -----------------------------
def draw_line(p1, p2, **kwargs):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)

# -----------------------------
# Coordinates
# -----------------------------
origin = np.array([0.0, 0.0])

# Gray coordinate axes
x1, x2 = np.array([-4.0, 0.0]), np.array([4.5, 0.0])
y1, y2 = np.array([0.0, -3.0]), np.array([0.0, 3.0])

# -----------------------------------------------
# MANUEL AYARLAR  (hepsi derece cinsinden)
disk_angle_deg  = 195    # 3. eksenin açısı (x eksenine göre)
arrow_angle_deg = 50   # okun yönü
ellipse_rot_deg = -30   # elipsin açısı
# -----------------------------------------------

disk_dir = np.array([np.cos(np.deg2rad(disk_angle_deg)),  np.sin(np.deg2rad(disk_angle_deg))])
nu_dir   = np.array([np.cos(np.deg2rad(arrow_angle_deg)), np.sin(np.deg2rad(arrow_angle_deg))])

# -----------------------------
# Main gray axes
# -----------------------------
axis_style = dict(color="0.65", lw=5, solid_capstyle="round", zorder=3)

draw_line(x1, x2, **axis_style)
draw_line(y1, y2, **axis_style)

# tilted gray plane line
p1 = origin - 4.1 * disk_dir
p2 = origin + 4.1 * disk_dir
draw_line(p1, p2, color="0.65", lw=5, solid_capstyle="round", zorder=2)

# -----------------------------
# Shaded disk / ellipse
# -----------------------------
ellipse_angle = ellipse_rot_deg

ellipse = Ellipse(
    xy=origin,
    width=6.9,
    height=1.45,
    angle=ellipse_angle,
    facecolor="0.85",
    edgecolor="0.20",
    lw=2.2,
    ls=(0, (3, 3)),
    alpha=0.45,
    zorder=1
)
ax.add_patch(ellipse)

# Add a second dashed outline darker for clarity
ellipse_outline = Ellipse(
    xy=origin,
    width=6.9,
    height=1.45,
    angle=ellipse_angle,
    facecolor="none",
    edgecolor="0.20",
    lw=2.2,
    ls=(0, (3, 3)),
    zorder=4
)
ax.add_patch(ellipse_outline)

# -----------------------------
# Neutrino direction arrow from origin
# -----------------------------
arrow_start = origin
arrow_end = origin + 1.55 * nu_dir

arrow = FancyArrowPatch(
    arrow_start,
    arrow_end,
    arrowstyle="-|>",
    mutation_scale=22,
    lw=2.2,
    color="black",
    zorder=6
)
ax.add_patch(arrow)

# -----------------------------
# 90 degree marker
# -----------------------------
# small square between disk direction and neutrino direction
s = 0.28
a = disk_dir
b = nu_dir

corner0 = origin + 0.18 * a
corner1 = corner0 + s * a
corner2 = corner1 + s * b
corner3 = corner0 + s * b

right_angle = Polygon(
    [corner0, corner1, corner2, corner3],
    closed=False,
    fill=False,
    edgecolor="black",
    lw=2.2,
    joinstyle="miter",
    zorder=7
)
ax.add_patch(right_angle)

# -----------------------------
# Optional label
# -----------------------------
label_pos = origin + 1.75 * nu_dir + np.array([0.35, 0.05])
ax.text(
    label_pos[0],
    label_pos[1],
    "neutrino direction",
    fontsize=16,
    fontfamily="serif",
    ha="left",
    va="center",
    color="black",
    zorder=8
)

# -----------------------------
# Limits and save
# -----------------------------
ax.set_xlim(-4.8, 5.0)
ax.set_ylim(-3.4, 3.4)

plt.tight_layout()
plt.savefig("neutrino_disk_diagram.png", dpi=300, bbox_inches="tight")
plt.savefig("neutrino_disk_diagram.svg", bbox_inches="tight")
plt.show()