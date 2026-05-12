import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------
# Surface function
# -----------------------------
def f(x, y):
    hill = 1.2 * np.exp(-((x - 0.6)**2 + (y - 0.4)**2) / 0.5)
    valley = 1.4 * np.exp(-((x + 0.4)**2 + (y + 0.6)**2) / 0.7)
    return hill - valley

# -----------------------------
# Grid
# -----------------------------
x = np.linspace(-1.8, 1.8, 120)
y = np.linspace(-1.8, 1.8, 120)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# -----------------------------
# Figure
# -----------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# surface
ax.plot_surface(
    X, Y, Z,
    cmap='jet',
    edgecolor='k',
    linewidth=0.6,
    antialiased=True,
    alpha=0.95
)

# -----------------------------
# Curved black path / arrow
# -----------------------------
# Start near the hill, end near the valley
t = np.linspace(0, 1, 80)

x_path = 0.55 - 0.95 * t + 0.10 * np.sin(np.pi * t)
y_path = 0.55 - 1.10 * t + 0.18 * np.sin(np.pi * t / 1.2)
z_path = f(x_path, y_path) + 0.03  # slightly above surface

# path line
ax.plot(x_path, y_path, z_path, color='black', linewidth=4)

# start point
ax.scatter(
    [x_path[0]], [y_path[0]], [z_path[0]],
    color='black', s=180, depthshade=False
)

# arrow head using last segment
dx = x_path[-1] - x_path[-4]
dy = y_path[-1] - y_path[-4]
dz = z_path[-1] - z_path[-4]

ax.quiver(
    x_path[-4], y_path[-4], z_path[-4],
    dx, dy, dz,
    color='black',
    linewidth=3,
    arrow_length_ratio=0.45
)

# -----------------------------
# View / style
# -----------------------------
ax.view_init(elev=22, azim=-125)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_zlim(Z.min() - 0.2, Z.max() + 0.2)

# tick style
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_zticks([])

# remove axis labels if you want screenshot-like look
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")

# grey panes like in the example
ax.xaxis.set_pane_color((0.94, 0.94, 0.94, 1.0))
ax.yaxis.set_pane_color((0.94, 0.94, 0.94, 1.0))
ax.zaxis.set_pane_color((0.94, 0.94, 0.94, 1.0))

# dotted grid feel
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis._axinfo["grid"]['linestyle'] = ':'
    axis._axinfo["grid"]['linewidth'] = 1.2
    axis._axinfo["grid"]['color'] = (0.6, 0.6, 0.6, 1)

plt.tight_layout()
plt.show()