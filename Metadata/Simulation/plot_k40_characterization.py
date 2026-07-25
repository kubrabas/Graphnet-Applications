#!/usr/bin/env python3
"""Plot the K40 characterization used by the P-ONE offline simulation."""

from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "pone_offline"
    / "NoiseGenerators"
    / "k40-characterization.hdf5"
)
MULTIPLICITY_OUTPUT = SCRIPT_DIR / "k40_event_multiplicity.pdf"
TIMING_OUTPUT = SCRIPT_DIR / "k40_multifold_time_offsets.pdf"
COINCIDENCE_OUTPUT = SCRIPT_DIR / "k40_pmt_coincidence_matrix.pdf"

# Geometry used by NoiseGenerators/K40Noise.py through Utilities/POMModel.py.
# Indices in the HDF5 file are zero-based; figure labels below are one-based.
PMT_ANGLES = np.array(
    [
        [57.5, 270.0],
        [57.5, 0.0],
        [57.5, 90.0],
        [57.5, 180.0],
        [25.0, 225.0],
        [25.0, 315.0],
        [25.0, 45.0],
        [25.0, 135.0],
        [-57.5, 270.0],
        [-57.5, 180.0],
        [-57.5, 90.0],
        [-57.5, 0.0],
        [-25.0, 315.0],
        [-25.0, 225.0],
        [-25.0, 135.0],
        [-25.0, 45.0],
    ]
)
UPPER_HOME_INDEX = 0
LOWER_HOME_INDEX = 4


def read_characterization(path):
    """Read the tables and rates consumed by K40Noise."""
    with h5py.File(path, "r") as h5_file:
        group = h5_file["coincidence-combinations"]
        combinations = group["combinations"][:]
        combination_weights = group["weights"][:]
        arrival_times = h5_file["arrival-times"][:]
        single_rate = float(group.attrs["singlefold-rate"])
        multi_rate = float(group.attrs["multifold-rate"])

    return (
        combinations,
        combination_weights,
        arrival_times,
        single_rate,
        multi_rate,
    )


def full_multiplicity_distribution(
    combinations,
    combination_weights,
    single_rate,
    multi_rate,
):
    """Combine the single-fold rate with the conditional multifold table."""
    weights = combination_weights / np.sum(combination_weights)
    stored_multiplicities = np.sum(combinations >= 0, axis=1)

    total_rate = single_rate + multi_rate
    single_fraction = single_rate / total_rate
    multi_fraction = multi_rate / total_rate

    multiplicities = np.arange(1, 6)
    probabilities = np.zeros_like(multiplicities, dtype=float)
    probabilities[0] = single_fraction

    for multiplicity in np.unique(stored_multiplicities):
        conditional_probability = np.sum(
            weights[stored_multiplicities == multiplicity]
        )
        probabilities[multiplicity - 1] = (
            multi_fraction * conditional_probability
        )

    return (
        multiplicities,
        probabilities,
        total_rate,
        single_fraction,
        multi_fraction,
    )


def arrival_time_density(arrival_times):
    """Reproduce the normalization and within-bin interpretation in K40Noise."""
    centres = arrival_times[:, 0]
    probabilities = arrival_times[:, 1]
    probabilities = probabilities / np.sum(probabilities)

    bin_width = float(np.median(np.diff(centres)))
    edges = np.concatenate(
        ([centres[0] - bin_width / 2.0], centres + bin_width / 2.0)
    )
    density = probabilities / bin_width

    return edges, density


def transform_combination(combination, horizontal_flip, vertical_flip, rotation):
    """Apply the same random symmetries as K40Noise.distribute_pmts."""
    angles = PMT_ANGLES[combination].copy()

    if UPPER_HOME_INDEX in combination:
        home = PMT_ANGLES[UPPER_HOME_INDEX]
    elif LOWER_HOME_INDEX in combination:
        home = PMT_ANGLES[LOWER_HOME_INDEX]
    else:
        raise ValueError("K40 combination does not contain a home PMT")

    if horizontal_flip:
        azimuth_differences = angles[:, 1] - home[1]
        angles[:, 1] -= 2.0 * azimuth_differences
        angles[:, 1] %= 360.0

    if vertical_flip:
        angles[:, 0] *= -1.0

    angles[:, 1] -= 90.0 * rotation
    angles[:, 1] %= 360.0

    angle_sums = np.sum(PMT_ANGLES, axis=1)
    transformed = np.empty(len(combination), dtype=int)
    for index, angle_sum in enumerate(np.sum(angles, axis=1)):
        matches = np.flatnonzero(np.isclose(angle_sums, angle_sum))
        if len(matches) != 1:
            raise ValueError(f"Could not map transformed PMT angle sum {angle_sum}")
        transformed[index] = matches[0]

    return transformed


def pmt_coincidence_matrix(combinations, combination_weights):
    """Expected PMT-pair inclusion probabilities for a multifold event."""
    weights = combination_weights / np.sum(combination_weights)
    matrix = np.zeros((len(PMT_ANGLES), len(PMT_ANGLES)), dtype=float)
    symmetry_weight = 1.0 / 16.0

    for padded_combination, combination_weight in zip(combinations, weights):
        combination = padded_combination[padded_combination >= 0]

        for horizontal_flip in (False, True):
            for vertical_flip in (False, True):
                for rotation in range(4):
                    transformed = transform_combination(
                        combination,
                        horizontal_flip,
                        vertical_flip,
                        rotation,
                    )
                    event_weight = combination_weight * symmetry_weight

                    for first_index in range(len(transformed)):
                        for second_index in range(first_index + 1, len(transformed)):
                            first_pmt = transformed[first_index]
                            second_pmt = transformed[second_index]
                            matrix[first_pmt, second_pmt] += event_weight
                            matrix[second_pmt, first_pmt] += event_weight

    return matrix


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


def make_multiplicity_figure(multiplicities, probabilities):
    """Suggested subfigure (a): PMT-hit multiplicity per K40 event."""
    configure_style()
    figure, multiplicity_axis = plt.subplots(
        figsize=(5.4, 4.2),
        constrained_layout=True,
    )
    multiplicity_axis.bar(
        multiplicities,
        probabilities,
        width=0.68,
        color="#0072B2",
        edgecolor="black",
        linewidth=0.7,
    )
    multiplicity_axis.set_yscale("log")
    multiplicity_axis.set_xlim(0.5, 5.5)
    multiplicity_axis.set_ylim(1e-9, 4.0)
    multiplicity_axis.set_xticks(multiplicities)
    multiplicity_axis.set_xlabel(
        r"Number of hit PMTs per $^{40}\mathrm{K}$ event"
    )
    multiplicity_axis.set_ylabel("Probability per event")
    multiplicity_axis.yaxis.set_major_locator(LogLocator(base=10))
    multiplicity_axis.yaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
    )
    multiplicity_axis.yaxis.set_minor_formatter(NullFormatter())
    multiplicity_axis.grid(
        axis="y",
        which="major",
        color="0.85",
        linewidth=0.7,
    )
    multiplicity_axis.set_axisbelow(True)

    for multiplicity, probability in zip(multiplicities, probabilities):
        percentage = 100.0 * probability
        if percentage >= 1.0:
            label = f"{percentage:.2f}%"
        elif percentage >= 0.01:
            label = f"{percentage:.3f}%"
        else:
            label = f"{percentage:.2e}%"

        multiplicity_axis.text(
            multiplicity,
            probability * 1.30,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

    return figure


def make_timing_figure(time_edges, time_density):
    """Suggested subfigure (b): relative hit-time offsets in multifold events."""
    configure_style()
    figure, timing_axis = plt.subplots(
        figsize=(5.4, 4.2),
        constrained_layout=True,
    )
    timing_axis.stairs(
        time_density,
        time_edges,
        fill=True,
        color="#009E73",
        alpha=0.30,
        linewidth=0.0,
    )
    timing_axis.stairs(
        time_density,
        time_edges,
        color="#007F5F",
        linewidth=1.8,
    )
    timing_axis.set_xlim(time_edges[0], time_edges[-1])
    timing_axis.set_ylim(bottom=0)
    timing_axis.set_xlabel("Relative hit-time offset (ns)")
    timing_axis.set_ylabel(r"Probability density (ns$^{-1}$)")
    timing_axis.grid(color="0.88", linewidth=0.7)
    timing_axis.set_axisbelow(True)

    return figure


def make_coincidence_figure(coincidence_matrix):
    """Optional subfigure (c): PMT pairs after random K40 symmetry operations."""
    configure_style()
    figure, axis = plt.subplots(
        figsize=(6.2, 5.2),
        constrained_layout=True,
    )

    display_matrix = coincidence_matrix.copy()
    np.fill_diagonal(display_matrix, np.nan)

    image = axis.imshow(
        display_matrix,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(16), labels=np.arange(1, 17))
    axis.set_yticks(np.arange(16), labels=np.arange(1, 17))
    axis.set_xlabel("PMT index")
    axis.set_ylabel("PMT index")
    axis.tick_params(top=False, right=False)

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Pair-inclusion probability per multifold event")

    return figure


def main():
    (
        combinations,
        combination_weights,
        arrival_times,
        single_rate,
        multi_rate,
    ) = read_characterization(INPUT_PATH)

    (
        multiplicities,
        probabilities,
        total_rate,
        single_fraction,
        multi_fraction,
    ) = full_multiplicity_distribution(
        combinations,
        combination_weights,
        single_rate,
        multi_rate,
    )

    time_edges, time_density = arrival_time_density(arrival_times)
    coincidence_matrix = pmt_coincidence_matrix(
        combinations,
        combination_weights,
    )

    multiplicity_figure = make_multiplicity_figure(
        multiplicities,
        probabilities,
    )
    multiplicity_figure.savefig(MULTIPLICITY_OUTPUT)
    plt.close(multiplicity_figure)

    timing_figure = make_timing_figure(time_edges, time_density)
    timing_figure.savefig(TIMING_OUTPUT)
    plt.close(timing_figure)

    coincidence_figure = make_coincidence_figure(coincidence_matrix)
    coincidence_figure.savefig(COINCIDENCE_OUTPUT)
    plt.close(coincidence_figure)

    print(f"Saved: {MULTIPLICITY_OUTPUT}")
    print(f"Saved: {TIMING_OUTPUT}")
    print(f"Saved: {COINCIDENCE_OUTPUT}")
    print(f"Total K40 event rate: {total_rate / 1e3:.6f} kHz")
    print(f"Single-fold fraction: {single_fraction:.8f}")
    print(f"Multifold fraction: {multi_fraction:.8f}")


if __name__ == "__main__":
    main()
