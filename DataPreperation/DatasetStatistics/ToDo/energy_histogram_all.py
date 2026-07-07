#!/usr/bin/env python3
"""
Usage:
    python3 DataPreperation/DatasetStatistics/energy_histogram_all.py \
  --mc-name STRING340MC \
  --geometry full_geometry

Plots energy distribution for all 4 flavors (Muon, Electron, Tau, NC)
as a 2x2 subplot.
"""

import argparse
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt

from icecube import icetray, dataio, dataclasses, LeptonInjector
icetray.I3Logger.global_logger = icetray.I3NullLogger()

sys.path.insert(0, "/project/def-nahee/kbas/Graphnet-Applications/Metadata")
import paths  # noqa: E402

# ── Dataset config ─────────────────────────────────────────────────────────────
MC_DATASETS = {
    "SPRING2026MC": paths.SPRING2026MC_I3,
    "STRING340MC": paths.STRING340MC_I3,
}

MC_FOLDERS = {
    "SPRING2026MC": "Spring2026MC",
    "STRING340MC": "340StringMC",
}

OUTPUT_BASE = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/DatasetStatistics"

COLORS = {
    "Muon":     "steelblue",
    "Electron": "tomato",
    "Tau":      "mediumseagreen",
    "NC":       "mediumpurple",
}

DECADE_EDGES = [2, 3, 4, 5, 6]
# ──────────────────────────────────────────────────────────────────────────────


def get_datasets(mc_name, geometry):
    return {
        flavor: info
        for flavor, info in MC_DATASETS[mc_name][geometry].items()
        if info.get("path") is not None and info.get("format") is not None
    }


def bad_file_rule(fpath):
    target = os.path.abspath(fpath)
    bad_i3_files = getattr(paths, "BAD_I3_FILES", {})

    for flavors in bad_i3_files.values():
        for info in flavors.values():
            if target in set(map(os.path.abspath, map(str, info.get("no_daq_for_some_reason", set())))):
                return "skip_no_daq", None

            available_daq_counts = info.get("available_daq_counts", {})
            for bad_path, daq_count in available_daq_counts.items():
                if target == os.path.abspath(str(bad_path)):
                    return "limit_daq", int(daq_count)

    return "normal", None


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
    total_skipped = 0
    total_bad = 0

    for fpath in files:
        rule, max_daq = bad_file_rule(fpath)
        if rule == "skip_no_daq":
            total_skipped += 1
            print(f"  [skip bad file] {fpath}")
            continue

        f = None
        daq_in_file = 0
        try:
            f = dataio.I3File(fpath)
            while f.more():
                frame = f.pop_frame()
                if frame.Stop != icetray.I3Frame.DAQ:
                    continue
                daq_in_file += 1
                if rule == "limit_daq" and daq_in_file > max_daq:
                    break
                total_daq += 1
                if "EventProperties" not in frame:
                    total_fail += 1
                    if rule == "limit_daq" and daq_in_file >= max_daq:
                        break
                    continue
                ep = frame["EventProperties"]
                energies.append(ep.totalEnergy)
                total_ok += 1
                if rule == "limit_daq" and daq_in_file >= max_daq:
                    break
        except Exception as e:
            total_bad += 1
            print(f"  [bad file] {fpath} -- {e}")
        finally:
            if f is not None:
                f.close()

    print(f"  DAQ frames : {total_daq}")
    print(f"  EP ok      : {total_ok}")
    print(f"  EP missing : {total_fail}")
    print(f"  Bad skipped: {total_skipped}")
    print(f"  Bad errors : {total_bad}")

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-name", default="SPRING2026MC", choices=list(MC_DATASETS))
    ap.add_argument("--geometry", default="full_geometry")
    args = ap.parse_args()

    DATASETS = get_datasets(args.mc_name, args.geometry)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (flavor, info) in zip(axes, DATASETS.items()):
        energies = collect_energies(flavor, info["path"], info["format"])
        if len(energies) == 0:
            ax.set_title(f"{flavor}  -  no data")
            continue
        plot_flavor(ax, flavor, energies)

    plt.suptitle(f"Energy distribution - {args.mc_name} ({args.geometry})", fontsize=13)
    plt.tight_layout()

    out_dir = os.path.join(OUTPUT_BASE, MC_FOLDERS[args.mc_name], args.geometry)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "energy_distribution_all.png")
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    plt.show()
