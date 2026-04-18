import argparse
import os
import re
from collections import defaultdict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from icecube import dataio


# ============================================================
# GCD file paths
# ============================================================

GCD_SPRING2026MC = (
    "/project/6008051/pone_simulation/GCD_Library"
    "/PONE_800mGrid_40mSpacing_40OMstring.i3.gz"
)
GCD_340STRINGMC = (
    "/project/6008051/pone_simulation/GCD_Library"
    "/PONE_800mGrid.i3.gz"
)


# ============================================================
# Argument parsing (single-dash flags only)
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot baseline and fill strings from two sub-geometry CSVs.\n"
            "The GCD file is chosen automatically from the campaign folder name."
        )
    )
    parser.add_argument(
        "csv1",
        help="Baseline (original) layout CSV path.",
    )
    parser.add_argument(
        "csv2",
        help="Expanded layout CSV path (contains baseline + fill strings).",
    )
    parser.add_argument(
        "-name",
        default=None,
        metavar="STEM",
        help="Output filename stem (no extension). Default: csv1_vs_csv2.png",
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
# GCD selection from campaign folder name
# ============================================================

def resolve_gcd(csv_path):
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
# Parse string count from filename (number between two underscores)
# ============================================================

def parse_string_count(csv_path):
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    match = re.search(r"_(\d+)_", stem)
    if match:
        return int(match.group(1))
    return None


# ============================================================
# Read string IDs from flat comma-separated CSV
# ============================================================

def read_string_ids(csv_path):
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
# Read string positions from GCD file
# ============================================================

def read_geometry(gcd_path):
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

    return pd.DataFrame(rows)


# ============================================================
# Plot
# ============================================================

def make_plot(
    baseline_df,
    fill_df,
    n_baseline,
    n_fill,
    csv1_stem,
    csv2_stem,
    campaign_label,
    out_path,
    dpi,
):
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.scatter(
        baseline_df["x"], baseline_df["y"],
        color="#1f77b4", s=50, zorder=3,
        label=f"{n_baseline} original  [{csv1_stem}]",
    )
    ax.scatter(
        fill_df["x"], fill_df["y"],
        color="#d62728", s=50, zorder=4,
        label=f"{n_fill} ROV fill  [{csv2_stem}, not in baseline]",
    )

    for _, row in pd.concat([baseline_df, fill_df]).iterrows():
        ax.text(
            float(row["x"]), float(row["y"]),
            str(int(row["string"])),
            fontsize=5, ha="left", va="bottom",
        )

    ax.set_title(
        f"Baseline and Fill String Layout\n{campaign_label}",
        fontsize=13,
    )
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal", adjustable="box")

    all_x = list(baseline_df["x"]) + list(fill_df["x"])
    all_y = list(baseline_df["y"]) + list(fill_df["y"])
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    xr = max(xmax - xmin, 1.0)
    yr = max(ymax - ymin, 1.0)
    pad = 0.04
    ax.set_xlim(xmin - pad * xr, xmax + pad * xr)
    ax.set_ylim(ymin - pad * yr, ymax + pad * yr)

    ax.legend(
        loc="upper left",
        fontsize=10,
        framealpha=0.9,
        title=f"Total: {n_baseline + n_fill} strings",
        title_fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    csv1_path = os.path.abspath(args.csv1)
    csv2_path = os.path.abspath(args.csv2)

    if os.path.dirname(csv1_path) != os.path.dirname(csv2_path):
        raise ValueError("Both CSVs must be in the same folder.")

    csv1_stem = os.path.splitext(os.path.basename(csv1_path))[0]
    csv2_stem = os.path.splitext(os.path.basename(csv2_path))[0]

    out_dir = os.path.dirname(csv1_path)
    out_name = args.name if args.name else f"{csv1_stem}_vs_{csv2_stem}"
    if not out_name.endswith(".png"):
        out_name += ".png"
    out_path = os.path.join(out_dir, out_name)

    gcd_path, campaign_label = resolve_gcd(csv1_path)

    n_csv1 = parse_string_count(csv1_path)
    n_csv2 = parse_string_count(csv2_path)

    print(f"Campaign:  {campaign_label}")
    print(f"GCD:       {gcd_path}")
    print(f"CSV 1 ({n_csv1} strings): {csv1_path}")
    print(f"CSV 2 ({n_csv2} strings): {csv2_path}")

    baseline_ids = set(read_string_ids(csv1_path))
    expanded_ids = set(read_string_ids(csv2_path))
    fill_ids = expanded_ids - baseline_ids

    print(f"Baseline strings: {len(baseline_ids)}")
    print(f"Fill strings (CSV2 minus CSV1): {len(fill_ids)}")

    full_geo = read_geometry(gcd_path)

    baseline_df = full_geo[full_geo["string"].isin(baseline_ids)].copy()
    fill_df = full_geo[full_geo["string"].isin(fill_ids)].copy()

    missing = sorted((baseline_ids | fill_ids) - set(full_geo["string"]))
    if missing:
        print(f"Warning: {len(missing)} ID(s) not found in GCD: {missing}")

    if baseline_df.empty and fill_df.empty:
        raise RuntimeError("No strings found to plot.")

    make_plot(
        baseline_df=baseline_df,
        fill_df=fill_df,
        n_baseline=len(baseline_df),
        n_fill=len(fill_df),
        csv1_stem=csv1_stem,
        csv2_stem=csv2_stem,
        campaign_label=campaign_label,
        out_path=out_path,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()


# ============================================================
# Usage examples
# ============================================================
#
# Basic:
#   python3 /project/def-nahee/kbas/Graphnet-Applications/DataPreperation/GeometryFiles/plot_rov_constraint_and_filled.py \
#       /project/def-nahee/kbas/.../Spring2026MC/strings_102_40m.csv \
#       /project/def-nahee/kbas/.../Spring2026MC/strings_157_40m.csv
#
# Custom output name and DPI:
#   python3 /project/def-nahee/kbas/Graphnet-Applications/DataPreperation/GeometryFiles/plot_rov_constraint_and_filled.py \
#       .../strings_102_40m.csv \
#       .../strings_157_40m.csv \
#       -name my_layout \
#       -dpi 300
