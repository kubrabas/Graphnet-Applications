#!/usr/bin/env python3
"""
Usage:
    python energy_histogram_all.py

Plots energy distribution for all 4 flavors (Muon, Electron, Tau, NC)
as a 2x2 subplot. Saves energy_distribution_all.png next to this script.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from icecube import icetray, dataio, dataclasses, LeptonInjector
icetray.I3Logger.global_logger = icetray.I3NullLogger()

# ── Dataset config ─────────────────────────────────────────────────────────────
DATASETS = {
    "Muon":     {"path": "/project/6008051/pone_simulation/MC000008-nu_mu-2_6-LeptonInjector_PROPOSAL_clsim-v17.1/Generator",    "format": "zst"},
    "Electron": {"path": "/project/6008051/pone_simulation/MC000009-nu_e-2_6-LeptonInjector_PROPOSAL_clsim-v17.1/Generator",    "format": "zst"},
    "Tau":      {"path": "/project/6008051/pone_simulation/MC000010-nu_tau-2_6-LeptonInjector_PROPOSAL_clsim-v17.1/Generator",   "format": "zst"},
    "NC":       {"path": "/project/6008051/pone_simulation/MC000011-nu_NC-2_6-LeptonInjector_PROPOSAL_clsim_NC-v17.1/Generator", "format": "zst"},
}

COLORS = {
    "Muon":     "steelblue",
    "Electron": "tomato",
    "Tau":      "mediumseagreen",
    "NC":       "mediumpurple",
}

DECADE_EDGES = [2, 3, 4, 5, 6]
# ──────────────────────────────────────────────────────────────────────────────


def collect_energies(flavor, path, fmt):
    pattern = "*.i3" if fmt == "i3" else f"*.i3.{fmt}"
    files = sorted(glob.glob(os.path.join(path, "**", pattern), recursive=True))

    if not files:
        print(f"[{flavor}] No files found under '{path}'")
        return np.array([])

    print(f"\n=== {flavor}: {len(files)} files ===")

    energies   = []
    total_daq  = 0
    total_ok   = 0
    total_fail = 0

    for fpath in files:
        f = dataio.I3File(fpath)
        while f.more():
            frame = f.pop_frame()
            if frame.Stop != icetray.I3Frame.DAQ:
                continue
            total_daq += 1
            if "EventProperties" not in frame:
                total_fail += 1
                continue
            ep = frame["EventProperties"]
            energies.append(ep.totalEnergy)
            total_ok += 1
        f.close()

    print(f"  DAQ frames : {total_daq}")
    print(f"  EP ok      : {total_ok}")
    print(f"  EP missing : {total_fail}")

    return np.array(energies)


def plot_flavor(ax, flavor, energies):
    color = COLORS[flavor]
    log_E = np.log10(energies)
    bins  = np.linspace(2, 6, 81)

    counts, edges = np.histogram(log_E, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ax.step(centers, counts, where="mid", linewidth=2, color=color)
    ax.fill_between(centers, counts, step="mid", alpha=0.15, color=color)

    total = len(energies)

    for i, (lo, hi) in enumerate(zip(DECADE_EDGES[:-1], DECADE_EDGES[1:])):
        if i > 0:
            ax.axvline(lo, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        mask = (log_E >= lo) & (log_E < hi)
        n    = mask.sum()
        pct  = 100 * n / total
        x_mid = (lo + hi) / 2

        ax.text(
            x_mid, counts.max() * 1.02,
            f"{n:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=8.5, color=color,
        )

    ax.set_xticks(DECADE_EDGES)
    ax.set_xticklabels(["100 GeV", "1 TeV", "10 TeV", "100 TeV", "1 PeV"], fontsize=9)
    ax.set_xlabel("Energy", fontsize=11)
    ax.set_ylabel("Event counts", fontsize=11)
    ax.set_title(f"{flavor}  -  {total:,} events", fontsize=11)
    ax.set_xlim(2, 6)
    ax.set_ylim(0, counts.max() * 1.18)


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (flavor, info) in zip(axes, DATASETS.items()):
        energies = collect_energies(flavor, info["path"], info["format"])
        if len(energies) == 0:
            ax.set_title(f"{flavor}  -  no data")
            continue
        plot_flavor(ax, flavor, energies)

    plt.suptitle("Energy distribution - MC000008-11  (Spring 2026, full geometry layout)", fontsize=13)
    plt.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "energy_distribution_all.png")
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    plt.show()