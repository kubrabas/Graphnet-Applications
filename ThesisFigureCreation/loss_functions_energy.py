import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 2.0,
    "figure.dpi": 150,
})

r = np.linspace(-3, 3, 1000)

mse      = r**2
mae      = np.abs(r)
logcosh  = np.log(np.cosh(r))

dmse     = 2 * r
dmae     = np.sign(r)
dlogcosh = np.tanh(r)

BLUE   = "#2166ac"
ORANGE = "#d6604d"
GREEN  = "#1a9641"

# ── plot 1: loss functions ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(r, mse,     color=BLUE,   linestyle="--", label=r"MSE: $r^2$")
ax.plot(r, mae,     color=ORANGE, linestyle=":",  label=r"MAE: $|r|$")
ax.plot(r, logcosh, color=GREEN,  linestyle="-",  label=r"LogCosh: $\log(\cosh(r))$", zorder=5)
ax.set_xlim(-3, 3)
ax.set_ylim(-0.1, 5.5)
ax.set_xlabel(r"$r = \hat{y} - y$")
ax.set_ylabel("Loss")
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
import os
_dir = os.path.dirname(os.path.abspath(__file__))
# fig.savefig(os.path.join(_dir, "loss_functions.pdf"), bbox_inches="tight", format="pdf")
fig.savefig(os.path.join(_dir, "loss_functions.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("saved: loss_functions")

# ── plot 2: gradients ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(r, dmse,     color=BLUE,   linestyle="--", label=r"MSE: $2r$")
ax.plot(r, dmae,     color=ORANGE, linestyle=":",  label=r"MAE: $\mathrm{sign}(r)$")
ax.plot(r, dlogcosh, color=GREEN,  linestyle="-",  label=r"LogCosh: $\tanh(r)$", zorder=5)
ax.set_xlim(-3, 3)
ax.set_ylim(-4.5, 4.5)
ax.set_xlabel(r"$r = \hat{y} - y$")
ax.set_ylabel("Gradient")
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.axhline( 1, color=ORANGE, linewidth=0.6, linestyle=":", alpha=0.4)
ax.axhline(-1, color=ORANGE, linewidth=0.6, linestyle=":", alpha=0.4)
ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
# fig.savefig(os.path.join(_dir, "loss_gradients.pdf"), bbox_inches="tight", format="pdf")
fig.savefig(os.path.join(_dir, "loss_gradients.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("saved: loss_gradients")