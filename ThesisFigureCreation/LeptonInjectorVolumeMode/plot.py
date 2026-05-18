import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path

fig, ax = plt.subplots(figsize=(7, 6))

ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")

# Cylinder parameters
cx = 5.0
top_y = 6.2
bottom_y = 2.0
width = 6.2
height = 1.0

# Cylinder edges
top = Ellipse((cx, top_y), width, height, fill=False, edgecolor="black", lw=1.2)

ax.add_patch(top)

# Bottom ellipse: back half dashed, visible front half solid.
theta_back = np.linspace(0, np.pi, 120)
theta_front = np.linspace(np.pi, 2 * np.pi, 120)
for theta, linestyle in ((theta_back, (0, (6, 4))), (theta_front, "-")):
    ax.plot(
        cx + width / 2 * np.cos(theta),
        bottom_y + height / 2 * np.sin(theta),
        color="black",
        lw=1.2,
        linestyle=linestyle,
    )

# Side lines
ax.plot([cx - width / 2, cx - width / 2], [bottom_y, top_y], color="black", lw=1.0, alpha=0.6)
ax.plot([cx + width / 2, cx + width / 2], [bottom_y, top_y], color="black", lw=1.0, alpha=0.6)

# Main neutrino trajectory
# Keep every point on the same line so the trajectory does not kink.
p_entry = np.array([cx - width / 2, 2.7])
p_exit = np.array([cx + width / 2, 5.2])
trajectory_slope = (p_exit[1] - p_entry[1]) / (p_exit[0] - p_entry[0])

def trajectory_y(x):
    return p_entry[1] + trajectory_slope * (x - p_entry[0])

p0 = np.array([0.45, trajectory_y(0.45)])
r_i = np.array([5.4, trajectory_y(5.4)])
p1 = np.array([9.75, trajectory_y(9.75)])

# Black outside line segments
ax.plot([p0[0], p_entry[0]], [p0[1], p_entry[1]], color="black", lw=2.4)
ax.plot([p_exit[0], p1[0]], [p_exit[1], p1[1]], color="black", lw=2.4)

# Orange effective segment
ax.plot(
    [p_entry[0], p_exit[0]],
    [p_entry[1], p_exit[1]],
    color="#ff7f0e",
    lw=3.2
)

# Points
ax.scatter(*p_entry, s=70, color="black", zorder=5)
ax.scatter(*r_i, s=70, color="black", zorder=5)
ax.scatter(*p_exit, s=70, color="black", zorder=5)

# Labels
ax.text(p_entry[0] - 1.15, p_entry[1] + 0.25, r"$p_{\mathrm{entry}}$", fontsize=20)
ax.text(p_exit[0] + 0.45, p_exit[1] - 0.2, r"$p_{\mathrm{exit}}$", fontsize=20)

ax.annotate(
    r"interaction vertex $\vec{v}$",
    xy=r_i,
    xytext=(3.9, 4.95),
    fontsize=14,
    arrowprops=dict(arrowstyle="-", lw=1.0),
    ha="left"
)

ax.text(
    4.45, 3.2,
    r"$L_{\mathrm{eff}} = |\vec{p}_{\mathrm{exit}} - \vec{p}_{\mathrm{entry}}|$",
    fontsize=16,
    color="#ff7f0e",
    rotation=25
)

plt.tight_layout()
plt.savefig(Path(__file__).with_suffix(".svg"), format="svg", bbox_inches="tight")
plt.show()
