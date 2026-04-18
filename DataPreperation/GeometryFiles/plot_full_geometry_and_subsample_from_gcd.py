import argparse
import os
from collections import defaultdict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from icecube import dataio


# ============================================================
# GCD file lookup based on campaign folder name in the csv path
# ============================================================

GCD_SPRING2026MC = (
    "/project/6008051/pone_simulation/GCD_Library"
    "/PONE_800mGrid_40mSpacing_40OMstring.i3.gz"
)
GCD_340STRINGMC = (
    "/project/6008051/pone_simulation/GCD_Library"
    "/PONE_800mGrid.i3.gz"
)


def resolve_gcd(csv_path):
    """Pick GCD file from the campaign name found in the csv path."""
    norm = csv_path.replace("\\", "/")
    if "Spring2026MC" in norm:
        return GCD_SPRING2026MC, "Spring 2026 MC"
    if "340StringMC" in norm:
        return GCD_340STRINGMC, "340 String MC"
    raise ValueError(
        f"Cannot determine campaign from path: {csv_path}\n"
        "Expected 'Spring2026MC' or '340StringMC' in the path."
    )


# ============================================================
# Argument parsing (single-dash flags only)
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot full GCD geometry alongside a sub-geometry from a CSV.\n"
            "The GCD file is chosen automatically from the campaign folder\n"
            "name embedded in the csv path."
        )
    )
    parser.add_argument(
        "csv_path",
        help=(
            "Path to the sub-geometry CSV file. The .csv extension is optional.\n"
            "Example: /project/.../Metadata/GeometryFiles/Spring2026MC/strings_102_40m"
        ),
    )
    parser.add_argument(
        "-out",
        default=None,
        metavar="DIR",
        help="Output directory. Defaults to the directory of the CSV file.",
    )
    parser.add_argument(
        "-name",
        default=None,
        metavar="STEM",
        help="Output filename stem, e.g. 'my_plot' saves as my_plot.png.",
    )
    parser.add_argument(
        "-dpi",
        type=int,
        default=200,
        metavar="DPI",
        help="Output image DPI (default: 200).",
    )
    return parser.parse_args()


# ============================================================
# CSV reader
# ============================================================

def read_string_ids(csv_path):
    """Read comma-separated integer string IDs from a flat CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, "r") as fh:
        content = fh.read().strip()

    if not content:
        raise ValueError(f"CSV is empty: {csv_path}")

    tokens = []
    for line in content.splitlines():
        for item in line.split(","):
            item = item.strip()
            if item:
                tokens.append(item)

    try:
        ids = sorted(set(int(x) for x in tokens))
    except ValueError as exc:
        raise ValueError(f"Non-integer value in {csv_path}") from exc

    if not ids:
        raise ValueError(f"No valid string IDs found in {csv_path}")

    return ids


# ============================================================
# GCD reader
# ============================================================

def read_geometry(gcd_path):
    """Return a DataFrame with columns [string, x, y] from the GCD file."""
    if not os.path.exists(gcd_path):
        raise FileNotFoundError(f"GCD file not found: {gcd_path}")

    f = dataio.I3File(gcd_path, "r")
    geo_frame = None
    while f.more():
        fr = f.pop_frame()
        if fr.Stop == fr.Geometry:
            geo_frame = fr
            break
    f.close()

    if geo_frame is None:
        raise RuntimeError(f"No Geometry frame found in {gcd_path}")

    om_map = geo_frame["I3Geometry"].omgeo
    print(f"OMGeoMap size: {len(om_map)}")

    xy_by_string = defaultdict(list)
    for omkey, omgeo in om_map.items():
        s = int(omkey.string)
        pos = omgeo.position
        xy_by_string[s].append((float(pos.x), float(pos.y)))

    rows = []
    for s, xys in sorted(xy_by_string.items()):
        rows.append({"string": s, "x": xys[0][0], "y": xys[0][1]})

    df = pd.DataFrame(rows)
    print(f"Total strings in GCD: {len(df)}")
    return df


# ============================================================
# Plotting
# ============================================================

def add_string_count(ax, n):
    ax.text(
        0.02, 0.98,
        f"String count: {n}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def set_equal_limits(ax, xs, ys, pad=0.03):
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xr = max(xmax - xmin, 1.0)
    yr = max(ymax - ymin, 1.0)
    ax.set_xlim(xmin - pad * xr, xmax + pad * xr)
    ax.set_ylim(ymin - pad * yr, ymax + pad * yr)


def make_side_by_side_plot(full_df, sub_df, campaign_label, csv_stem, out_path, dpi):
    """
    Left panel:  sub-geometry strings only (black).
    Right panel: full geometry (black) with sub-geometry strings highlighted in red.
    """
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    # ---- left: sub-geometry only ----
    ax_left = axes[0]
    ax_left.scatter(sub_df["x"], sub_df["y"], color="black", s=15, zorder=2)
    for _, row in sub_df.iterrows():
        ax_left.text(
            float(row["x"]), float(row["y"]),
            str(int(row["string"])),
            fontsize=5, ha="left", va="bottom",
        )
    ax_left.set_title(f"Sub-geometry: {csv_stem}\n({campaign_label})")
    ax_left.set_xlabel("X (meters)")
    ax_left.set_ylabel("Y (meters)")
    ax_left.set_aspect("equal", adjustable="box")
    set_equal_limits(ax_left, sub_df["x"].tolist(), sub_df["y"].tolist())
    add_string_count(ax_left, len(sub_df))

    # ---- right: full geometry + sub-geometry highlighted ----
    ax_right = axes[1]

    not_selected = full_df[~full_df["string"].isin(sub_df["string"])]
    selected = full_df[full_df["string"].isin(sub_df["string"])]

    ax_right.scatter(
        not_selected["x"], not_selected["y"],
        color="#aaaaaa", s=15, zorder=2, label="Full geometry",
    )
    ax_right.scatter(
        selected["x"], selected["y"],
        color="red", s=25, zorder=3, label=f"Sub-geometry ({len(selected)} strings)",
    )
    ax_right.set_title(f"Full Geometry with Sub-geometry Highlighted\n({campaign_label})")
    ax_right.set_xlabel("X (meters)")
    ax_right.set_ylabel("Y (meters)")
    ax_right.set_aspect("equal", adjustable="box")
    set_equal_limits(ax_right, full_df["x"].tolist(), full_df["y"].tolist())
    ax_right.legend(loc="upper right", fontsize=9)
    add_string_count(ax_right, len(full_df))

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # Resolve CSV path (add .csv extension if missing)
    csv_path = args.csv_path
    if not csv_path.endswith(".csv"):
        csv_path = csv_path + ".csv"
    csv_path = os.path.abspath(csv_path)

    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]

    # Determine output location
    out_dir = args.out if args.out else os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    out_name = args.name if args.name else csv_stem
    if not out_name.endswith(".png"):
        out_name = out_name + ".png"
    out_path = os.path.join(out_dir, out_name)

    # Detect campaign and GCD
    gcd_path, campaign_label = resolve_gcd(csv_path)
    print(f"Campaign:  {campaign_label}")
    print(f"GCD file:  {gcd_path}")
    print(f"CSV file:  {csv_path}")

    # Read data
    selected_ids = read_string_ids(csv_path)
    print(f"Selected strings in CSV: {len(selected_ids)}")

    full_df = read_geometry(gcd_path)

    sub_df = full_df[full_df["string"].isin(selected_ids)].copy()

    missing = sorted(set(selected_ids) - set(sub_df["string"].tolist()))
    if missing:
        print(f"Warning: {len(missing)} string ID(s) from CSV not found in GCD: {missing}")

    if sub_df.empty:
        raise RuntimeError("No matching strings found between CSV and GCD.")

    make_side_by_side_plot(full_df, sub_df, campaign_label, csv_stem, out_path, args.dpi)


if __name__ == "__main__":
    main()


# ============================================================
# Usage examples
# ============================================================
#
# Spring 2026 MC sub-geometry:
#   python3 /project/def-nahee/kbas/Graphnet-Applications/DataPreperation/GeometryFiles/plot_full_geometry_and_subsample_from_gcd.py \
#       /project/def-nahee/kbas/Graphnet-Applications/Metadata/GeometryFiles/Spring2026MC/strings_102_40m
#
# 340 String MC sub-geometry with custom output name and DPI:
#   python3 /project/def-nahee/kbas/Graphnet-Applications/DataPreperation/GeometryFiles/plot_full_geometry_and_subsample_from_gcd.py \
#       /project/def-nahee/kbas/Graphnet-Applications/Metadata/GeometryFiles/340StringMC/strings_340 \
#       -name my_output \
#       -dpi 300
#
# Save to a different directory:
#   python3 /project/def-nahee/kbas/Graphnet-Applications/DataPreperation/GeometryFiles/plot_full_geometry_and_subsample_from_gcd.py \
#       /project/def-nahee/kbas/Graphnet-Applications/Metadata/GeometryFiles/Spring2026MC/strings_102_40m \
#       -out /project/def-nahee/kbas/plots
