#!/usr/bin/env python3
"""Plot mean accepted pulses per active PMT per event in energy bins."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_STATISTICS_DIR = SCRIPT_DIR.parent / "DatasetStatistics"
TRUTH_DIR = DATASET_STATISTICS_DIR / "TruthLevelStatistics"
FEATURE_DIR = DATASET_STATISTICS_DIR / "FeatureLevelStatistics"
DEFAULT_OUTPUT = SCRIPT_DIR / "accepted_pulses_per_active_pmt_per_event_vs_energy.pdf"
FLAVORS = ("Electron", "Muon", "NC", "Tau")
LAYOUTS = ("340", "160", "102")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flavors",
        nargs="+",
        choices=FLAVORS,
        default=list(FLAVORS),
    )
    parser.add_argument("--energy-min", type=float, default=1.0e2, help="GeV")
    parser.add_argument("--energy-max", type=float, default=1.0e7, help="GeV")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_energy_lookup(path):
    """Load energies into a compact array indexed by (RunID, EventID)."""
    maximum_run_id = 0
    maximum_event_id = 0
    with path.open("r", newline="") as truth_file:
        for row in csv.DictReader(truth_file):
            maximum_run_id = max(maximum_run_id, int(row["RunID"]))
            maximum_event_id = max(maximum_event_id, int(row["EventID"]))

    energies = np.full((maximum_run_id + 1, maximum_event_id + 1), np.nan)
    with path.open("r", newline="") as truth_file:
        for row_number, row in enumerate(csv.DictReader(truth_file), start=2):
            run_id = int(row["RunID"])
            event_id = int(row["EventID"])
            if np.isfinite(energies[run_id, event_id]):
                raise ValueError(
                    f"Duplicate (RunID, EventID)=({run_id}, {event_id}) "
                    f"in {path} at row {row_number}"
                )
            energies[run_id, event_id] = float(row["totalEnergy"])

    return energies


def accumulate_flavor(flavor, bin_edges):
    """Accumulate counts and pulse sums while validating row-by-row alignment."""
    truth_path = TRUTH_DIR / f"{flavor}.csv"
    feature_path = FEATURE_DIR / f"{flavor}.csv"
    number_of_bins = len(bin_edges) - 1
    active_event_counts = {
        layout: np.zeros(number_of_bins, dtype=np.int64) for layout in LAYOUTS
    }
    ratio_sums = {layout: np.zeros(number_of_bins) for layout in LAYOUTS}
    excluded_events = 0

    energies = load_energy_lookup(truth_path)
    matched_events = 0
    with feature_path.open("r", newline="") as feature_file:
        for row_number, feature_row in enumerate(
            csv.DictReader(feature_file), start=2
        ):
            run_id = int(feature_row["RunID"])
            event_id = int(feature_row["EventID"])
            if run_id >= energies.shape[0] or event_id >= energies.shape[1]:
                raise ValueError(
                    f"No truth energy for (RunID, EventID)=({run_id}, {event_id}) "
                    f"in {flavor} feature row {row_number}"
                )
            energy = energies[run_id, event_id]
            if not np.isfinite(energy):
                raise ValueError(
                    f"No truth energy for (RunID, EventID)=({run_id}, {event_id}) "
                    f"in {flavor} feature row {row_number}"
                )
            matched_events += 1
            bin_index = int(np.searchsorted(bin_edges, energy, side="right") - 1)
            if bin_index < 0 or bin_index >= number_of_bins:
                excluded_events += 1
                continue

            for layout in LAYOUTS:
                pulses = float(feature_row[f"accepted_pulse_count_{layout}_string"])
                active_pmts = int(
                    feature_row[
                        f"accepted_pulse_map_distinct_PMT_count_{layout}_string"
                    ]
                )
                if pulses < 0 or active_pmts < 0:
                    raise ValueError(
                        f"Missing accepted-pulse data for {flavor} at row {row_number}"
                    )
                if active_pmts == 0:
                    if pulses != 0:
                        raise ValueError(
                            f"Positive pulse count with zero active PMTs for {flavor} "
                            f"at row {row_number}"
                        )
                    continue

                pulses_per_active_pmt = pulses / active_pmts
                active_event_counts[layout][bin_index] += 1
                ratio_sums[layout][bin_index] += pulses_per_active_pmt

    if matched_events != np.count_nonzero(np.isfinite(energies)):
        raise ValueError(
            f"Truth/feature event-count mismatch for {flavor}: "
            f"matched {matched_events}, truth has "
            f"{np.count_nonzero(np.isfinite(energies))}"
        )

    means = {}
    for layout in LAYOUTS:
        populated = active_event_counts[layout] > 0
        means[layout] = np.full(number_of_bins, np.nan)
        means[layout][populated] = (
            ratio_sums[layout][populated] / active_event_counts[layout][populated]
        )

    return active_event_counts, means, excluded_events


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.linewidth": 0.9,
            "savefig.bbox": "tight",
        }
    )


def make_figure(results, bin_edges):
    configure_style()
    flavors = list(results)
    number_of_columns = 2 if len(flavors) > 1 else 1
    number_of_rows = int(np.ceil(len(flavors) / number_of_columns))
    figure, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(6.8 * number_of_columns, 4.5 * number_of_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    styles = {
        "340": ("#0072B2", "o"),
        "160": ("#D55E00", "s"),
        "102": ("#009E73", "^"),
    }

    for axis, flavor in zip(axes.flat, flavors):
        _, means, _ = results[flavor]
        for layout in LAYOUTS:
            color, marker = styles[layout]
            axis.plot(
                centres,
                means[layout],
                color=color,
                marker=marker,
                markersize=4.0,
                linewidth=1.3,
                label=f"{layout}-string layout",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(flavor)
        axis.grid(which="both", color="0.88", linewidth=0.6)
        axis.set_axisbelow(True)

    for axis in axes.flat[len(flavors) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel(r"Neutrino energy $E_\nu$ [GeV]")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean accepted pulses per active PMT per event")
    axes.flat[0].legend(frameon=False)
    return figure


def main():
    args = parse_args()
    if args.energy_min <= 0 or args.energy_max <= args.energy_min:
        raise ValueError("Require 0 < energy-min < energy-max")
    if args.bins < 1:
        raise ValueError("--bins must be positive")

    bin_edges = np.geomspace(args.energy_min, args.energy_max, args.bins + 1)
    results = {
        flavor: accumulate_flavor(flavor, bin_edges) for flavor in args.flavors
    }
    figure = make_figure(results, bin_edges)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    plt.close(figure)

    print(f"Saved: {args.output}")
    for flavor, (active_event_counts, _, excluded_events) in results.items():
        active_summary = " ".join(
            f"active_{layout}={active_event_counts[layout].sum()}"
            for layout in LAYOUTS
        )
        print(
            f"{flavor}: {active_summary} "
            f"excluded_outside_energy_range={excluded_events}"
        )


if __name__ == "__main__":
    main()
