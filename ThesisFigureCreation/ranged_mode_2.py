import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os

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
# MANUAL SETTINGS  (all in degrees)
disk_angle_deg  = 195   # tilted gray axis
ellipse_rot_deg = -30   # ellipse rotation
dot_t           = -0.51   # angle on ellipse (radians, 0 = right tip)
dot_r           = 0.3   # radial position: 0 = center, 1 = edge
# -----------------------------------------------

disk_dir = np.array([
    np.cos(np.deg2rad(disk_angle_deg)),
    np.sin(np.deg2rad(disk_angle_deg))
])

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
ellipse_width  = 6.9
ellipse_height = 1.45
ellipse_angle  = ellipse_rot_deg

ellipse = Ellipse(
    xy=origin,
    width=ellipse_width,
    height=ellipse_height,
    angle=ellipse_angle,
    facecolor="0.85",
    edgecolor="0.20",
    lw=2.2,
    ls=(0, (3, 3)),
    alpha=0.45,
    zorder=1
)
ax.add_patch(ellipse)

ellipse_outline = Ellipse(
    xy=origin,
    width=ellipse_width,
    height=ellipse_height,
    angle=ellipse_angle,
    facecolor="none",
    edgecolor="0.20",
    lw=2.2,
    ls=(0, (3, 3)),
    zorder=4
)
ax.add_patch(ellipse_outline)

# -----------------------------
# Red dot on the ellipse
# -----------------------------
a_ell = ellipse_width / 2
b_ell = ellipse_height / 2
theta = np.deg2rad(ellipse_rot_deg)

# dot_t controls where on the ellipse edge the dot sits
dot_x = dot_r * (a_ell * np.cos(dot_t) * np.cos(theta) - b_ell * np.sin(dot_t) * np.sin(theta))
dot_y = dot_r * (a_ell * np.cos(dot_t) * np.sin(theta) + b_ell * np.sin(dot_t) * np.cos(theta))

ax.plot(dot_x, dot_y, "o", color="red", markersize=10, zorder=8)

# -----------------------------
# Limits and save
# -----------------------------
ax.set_xlim(-4.8, 5.0)
ax.set_ylim(-3.4, 3.4)

plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(script_dir, "ranged_mode_2.png"), dpi=300, bbox_inches="tight")
plt.show()
