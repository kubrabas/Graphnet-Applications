#!/usr/bin/env python3
"""
Thesis cross-section figures from CSMS model.
Reads photospline FITS files and produces cross-section plots saved in this folder.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from pathlib import Path
import photospline

DATA_DIR = Path(
    "/project/6008051/pone_simulation/pone_offline"
    "/CrossSectionModels/csms_differential_v1.0"
)
OUT_DIR = Path(__file__).parent
MAX_LOG10_ENERGY = 8.0
DIFFERENTIAL_LOG10_ENERGIES = [2, 3, 4, 5, 6]
NU_COLOR = "#0072B2"
NUBAR_COLOR = "#D55E00"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _eval_1d(spl, n=600):
    log_e_min, log_e_max = spl.extents[0]
    log_e = np.linspace(log_e_min, min(log_e_max, MAX_LOG10_ENERGY), n)
    log_s = spl.evaluate_simple([log_e])
    return 10**log_e, 10**log_s


def _eval_slice(spl, log_e_val, n=120):
    """Evaluate d²σ/dxdy on an (x, y) grid at a fixed log10(E)."""
    (x_lo, x_hi), (y_lo, y_hi) = spl.extents[1], spl.extents[2]
    x_ax = np.linspace(x_lo, x_hi, n)
    y_ax = np.linspace(y_lo, y_hi, n)
    X, Y = np.meshgrid(x_ax, y_ax, indexing="ij")
    E_ax = np.full(X.size, log_e_val)
    log_z = spl.evaluate_simple([E_ax, X.ravel(), Y.ravel()])
    z = 10 ** log_z.reshape(n, n)
    return x_ax, y_ax, z


def _style_log_axes(ax):
    ax.grid(True, which="major", alpha=0.28)
    ax.grid(True, which="minor", alpha=0.12)


def _annotate_energy(ax, log_e):
    ax.text(
        0.06,
        0.92,
        rf"$10^{{{log_e}}}$ GeV",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
    )


def _differential_norm(grids):
    pos_vals = np.concatenate([z[np.isfinite(z) & (z > 0)].ravel() for _, _, z in grids])
    return mcolors.LogNorm(
        vmin=np.percentile(pos_vals, 1),
        vmax=np.percentile(pos_vals, 99.5),
        clip=True,
    )


def _add_differential_colorbar(fig, axes, im):
    cbar = fig.colorbar(
        im,
        ax=axes,
        label=r"$d^2\sigma/dx\,dy\ [\mathrm{cm}^2]$",
        shrink=0.85,
    )
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    cbar.ax.yaxis.set_minor_locator(mticker.NullLocator())
    return cbar


# ── individual plot functions ─────────────────────────────────────────────────


def plot_sigma_nu(out_path: Path):
    spl = photospline.SplineTable(str(DATA_DIR / "sigma_nu_CC_iso.fits"))
    print(f"sigma_nu  extents: {spl.extents}")
    energy, sigma = _eval_1d(spl)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(energy, sigma / 1e-36, color=NU_COLOR, lw=2.2)
    ax.set_xlabel("Neutrino Energy [GeV]")
    ax.set_ylabel(r"$\sigma_{\nu,\,\mathrm{CC}}\ [10^{-36}\ \mathrm{cm}^2]$")
    _style_log_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_sigma_nubar(out_path: Path):
    spl = photospline.SplineTable(str(DATA_DIR / "sigma_nubar_CC_iso.fits"))
    print(f"sigma_nubar extents: {spl.extents}")
    energy, sigma = _eval_1d(spl)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(energy, sigma / 1e-36, color=NUBAR_COLOR, lw=2.2)
    ax.set_xlabel("Neutrino Energy [GeV]")
    ax.set_ylabel(
        r"$\sigma_{\bar\nu,\,\mathrm{CC}}\ [10^{-36}\ \mathrm{cm}^2]$"
    )
    _style_log_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_sigma_nu_and_nubar(out_path: Path):
    spl_nu = photospline.SplineTable(str(DATA_DIR / "sigma_nu_CC_iso.fits"))
    spl_nubar = photospline.SplineTable(str(DATA_DIR / "sigma_nubar_CC_iso.fits"))
    print(f"sigma_nu/nubar extents: {spl_nu.extents} / {spl_nubar.extents}")
    energy_nu, sigma_nu = _eval_1d(spl_nu)
    energy_nubar, sigma_nubar = _eval_1d(spl_nubar)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(energy_nu, sigma_nu / 1e-36, color=NU_COLOR, lw=2.2, label=r"$\nu$")
    ax.loglog(
        energy_nubar,
        sigma_nubar / 1e-36,
        color=NUBAR_COLOR,
        lw=2.2,
        label=r"$\bar\nu$",
    )
    ax.set_xlabel("Neutrino Energy [GeV]")
    ax.set_ylabel(r"$\sigma_{\mathrm{CC}}\ [10^{-36}\ \mathrm{cm}^2]$")
    _style_log_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_dsdxdy_nu(out_path: Path):
    spl = photospline.SplineTable(str(DATA_DIR / "dsdxdy_nu_CC_iso.fits"))
    print(f"dsdxdy_nu extents: {spl.extents}")

    log_e_min, log_e_max = spl.extents[0]
    log_e_vals = [v for v in DIFFERENTIAL_LOG10_ENERGIES if log_e_min <= v <= log_e_max]

    fig, axes = plt.subplots(
        1, len(log_e_vals),
        figsize=(3.6 * len(log_e_vals), 4.2),
        sharey=True,
        constrained_layout=True,
    )
    if len(log_e_vals) == 1:
        axes = [axes]

    all_grids = [_eval_slice(spl, e) for e in log_e_vals]
    norm = _differential_norm(all_grids)

    im = None
    for ax, log_e, (x_ax, y_ax, z) in zip(axes, log_e_vals, all_grids):
        im = ax.pcolormesh(x_ax, y_ax, z.T, norm=norm, cmap="plasma", shading="auto")
        _annotate_energy(ax, log_e)
        ax.set_xlabel(r"$\log_{10}(x)$" if x_ax[0] < 0 else r"$x_{\rm Bj}$")

    axes[0].set_ylabel(r"$\log_{10}(y)$")
    _add_differential_colorbar(fig, axes, im)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_dsdxdy_nubar(out_path: Path):
    spl = photospline.SplineTable(str(DATA_DIR / "dsdxdy_nubar_CC_iso.fits"))
    print(f"dsdxdy_nubar extents: {spl.extents}")

    log_e_min, log_e_max = spl.extents[0]
    log_e_vals = [v for v in DIFFERENTIAL_LOG10_ENERGIES if log_e_min <= v <= log_e_max]

    fig, axes = plt.subplots(
        1, len(log_e_vals),
        figsize=(3.6 * len(log_e_vals), 4.2),
        sharey=True,
        constrained_layout=True,
    )
    if len(log_e_vals) == 1:
        axes = [axes]

    all_grids = [_eval_slice(spl, e) for e in log_e_vals]
    norm = _differential_norm(all_grids)

    im = None
    for ax, log_e, (x_ax, y_ax, z) in zip(axes, log_e_vals, all_grids):
        im = ax.pcolormesh(x_ax, y_ax, z.T, norm=norm, cmap="plasma", shading="auto")
        _annotate_energy(ax, log_e)
        ax.set_xlabel(r"$\log_{10}(x)$" if x_ax[0] < 0 else r"$x_{\rm Bj}$")

    axes[0].set_ylabel(r"$\log_{10}(y)$")
    _add_differential_colorbar(fig, axes, im)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_dsdxdy_nu_and_nubar(out_path: Path):
    spl_nu = photospline.SplineTable(str(DATA_DIR / "dsdxdy_nu_CC_iso.fits"))
    spl_nubar = photospline.SplineTable(str(DATA_DIR / "dsdxdy_nubar_CC_iso.fits"))
    print(f"dsdxdy_nu/nubar extents: {spl_nu.extents} / {spl_nubar.extents}")

    log_e_min = max(spl_nu.extents[0][0], spl_nubar.extents[0][0])
    log_e_max = min(spl_nu.extents[0][1], spl_nubar.extents[0][1])
    log_e_vals = [v for v in DIFFERENTIAL_LOG10_ENERGIES if log_e_min <= v <= log_e_max]

    nu_grids = [_eval_slice(spl_nu, e) for e in log_e_vals]
    nubar_grids = [_eval_slice(spl_nubar, e) for e in log_e_vals]
    norm = _differential_norm(nu_grids + nubar_grids)

    fig, axes = plt.subplots(
        2,
        len(log_e_vals),
        figsize=(3.6 * len(log_e_vals), 7.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(log_e_vals) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    im = None
    for row_axes, grids in zip(axes, [nu_grids, nubar_grids]):
        for ax, log_e, (x_ax, y_ax, z) in zip(row_axes, log_e_vals, grids):
            im = ax.pcolormesh(
                x_ax, y_ax, z.T, norm=norm, cmap="plasma", shading="auto"
            )

    for ax, log_e in zip(axes[0], log_e_vals):
        ax.set_title(rf"$10^{{{log_e}}}$ GeV", pad=8)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\log_{10}(x)$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10}(y)$")
    for ax, row_label in zip(axes[:, 0], [r"$\nu$", r"$\bar\nu$"]):
        ax.text(
            -0.30,
            0.5,
            row_label,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=15,
            clip_on=False,
        )

    _add_differential_colorbar(fig, axes.ravel().tolist(), im)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_sigma_nu(OUT_DIR / "sigma_nu_CC_iso.png")
    plot_sigma_nubar(OUT_DIR / "sigma_nubar_CC_iso.png")
    plot_sigma_nu_and_nubar(OUT_DIR / "sigma_nu_nubar_CC_iso.png")
    plot_dsdxdy_nu(OUT_DIR / "dsdxdy_nu_CC_iso.png")
    plot_dsdxdy_nubar(OUT_DIR / "dsdxdy_nubar_CC_iso.png")
    plot_dsdxdy_nu_and_nubar(OUT_DIR / "dsdxdy_nu_nubar_CC_iso.png")
