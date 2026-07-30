#!/usr/bin/env python3
"""Compare simulated per-PMT dark-noise counts with a Poisson model."""

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_STATISTICS_DIR = SCRIPT_DIR.parent / "DatasetStatistics" / "FeatureLevelStatistics"
DEFAULT_INPUTS = [
    FEATURE_STATISTICS_DIR / f"{flavor}.csv"
    for flavor in ("Electron", "Muon", "NC", "Tau")
]
DEFAULT_OUTPUT = SCRIPT_DIR / "dark_noise_poisson_validation.pdf"
COUNT_COLUMN = "dark_noise_1_1_1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare observed dark-noise hit counts for one PMT with the "
            "expected Poisson distribution."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Feature-level CSV files (default: Electron, Muon, NC, and Tau)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rate-per-ns", type=float, default=1.0e-6)
    parser.add_argument("--window-start-ns", type=float, default=-2000.0)
    parser.add_argument("--window-end-ns", type=float, default=10000.0)
    return parser.parse_args()


def read_counts(paths):
    """Read valid dark-noise counts without loading the large CSVs into memory."""
    multiplicities = Counter()
    invalid_rows = 0

    for path in paths:
        with path.open("r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if COUNT_COLUMN not in (reader.fieldnames or []):
                raise ValueError(f"Missing column {COUNT_COLUMN!r} in {path}")

            for row in reader:
                count = int(row[COUNT_COLUMN])
                if count < 0:
                    invalid_rows += 1
                    continue
                multiplicities[count] += 1

    return multiplicities, invalid_rows


def poisson_pmf(values, mean):
    return np.array(
        [math.exp(-mean) * mean**int(value) / math.factorial(int(value)) for value in values]
    )


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": "tight",
        }
    )


def make_figure(multiplicities, poisson_mean):
    total = sum(multiplicities.values())
    maximum = max(3, max(multiplicities))
    values = np.arange(maximum + 1)
    observed = np.array([multiplicities[value] / total for value in values])
    expected = poisson_pmf(values, poisson_mean)

    configure_style()
    figure, axis = plt.subplots(figsize=(5.6, 4.3), constrained_layout=True)
    width = 0.36
    simulation_bars = axis.bar(
        values - width / 2,
        observed,
        width=width,
        color="#0072B2",
        edgecolor="black",
        linewidth=0.7,
        label=(
            r"$P_{\mathrm{sim}}(N=n)="
            r"\frac{N_n}{\sum_{m=0}^{\infty}N_m}$"
        ),
    )
    axis.bar(
        values + width / 2,
        expected,
        width=width,
        color="#D55E00",
        edgecolor="black",
        linewidth=0.7,
        label=rf"Poisson ($\mu={poisson_mean:.3f}$)",
    )
    axis.set_yscale("log")
    axis.set_xlim(-0.6, maximum + 0.6)
    axis.set_ylim(bottom=min(observed[observed > 0].min(), expected.min()) * 0.35, top=3.0)
    axis.set_xticks(values)
    axis.set_xlabel("Number of dark-noise photoelectrons per PMT")
    axis.set_ylabel("Probability per time window")
    axis.grid(axis="y", which="both", color="0.88", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)

    for value, bar in zip(values, simulation_bars):
        event_count = multiplicities[value]
        label_x = value - {2: 0.10, 3: 0.20}.get(value, 0.0)
        axis.annotate(
            rf"$N_{{{value}}}={event_count:,}$",
            (label_x, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#0072B2",
        )

    return figure


def main():
    args = parse_args()
    window_duration = args.window_end_ns - args.window_start_ns
    if window_duration <= 0:
        raise ValueError("The end of the noise window must be after its start")

    poisson_mean = args.rate_per_ns * window_duration
    multiplicities, invalid_rows = read_counts(args.inputs)
    if not multiplicities:
        raise ValueError("No valid dark-noise counts were found")

    total = sum(multiplicities.values())
    observed_mean = sum(count * frequency for count, frequency in multiplicities.items()) / total

    figure = make_figure(multiplicities, poisson_mean)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    plt.close(figure)

    print(f"Saved: {args.output}")
    print(f"Valid events: {total}")
    print(f"Excluded rows (count < 0): {invalid_rows}")
    print(f"Expected mean: {poisson_mean:.8f}")
    print(f"Observed mean: {observed_mean:.8f}")
    for count in sorted(multiplicities):
        print(f"N={count}: {multiplicities[count]} events")


if __name__ == "__main__":
    main()
