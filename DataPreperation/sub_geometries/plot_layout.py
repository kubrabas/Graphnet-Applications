import argparse
import os
from collections import defaultdict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from icecube import dataio


# ============================================================
# 0) SETTINGS
# ============================================================

# old gcd:
# gcd_path = "/project/6008051/pone_simulation/GCD_Library/PONE_800mGrid.i3.gz"
# new gcd:
gcd_path = "/project/6008051/pone_simulation/GCD_Library/PONE_800mGrid_40mSpacing_40OMstring.i3.gz"

campaign_folder = "spring_2026_mc_campaign"

# Which sub-geometry CSV to use
parser = argparse.ArgumentParser()
parser.add_argument("geometry_name", help="e.g. strings_102_40m")
args = parser.parse_args()
geometry_name = args.geometry_name

# Read CSV from the same folder as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, campaign_folder, f"{geometry_name}.csv")

# Output
out_png_sub = os.path.join(script_dir, campaign_folder, f"{geometry_name}.png")


# ============================================================
# 1) HELPER: READ STRING IDS FROM SIMPLE CSV
# ============================================================

def read_string_ids(csv_path):
    """
    Reads a CSV like:
        272,273,253,254,...
    or multi-line comma-separated values with no header.

    Returns:
        sorted unique list of integer string IDs
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    with open(csv_path, "r") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"{csv_path} is empty")

    tokens = []
    for line in content.splitlines():
        for item in line.split(","):
            item = item.strip()
            if item:
                tokens.append(item)

    try:
        string_ids = sorted(set(int(x) for x in tokens))
    except ValueError as e:
        raise ValueError(f"{csv_path} contains non-integer values") from e

    if not string_ids:
        raise ValueError(f"No valid string IDs found in {csv_path}")

    return string_ids


# ============================================================
# 2) HELPER: PLOT STRINGS
# ============================================================

def plot_strings(df, title, out_path, count_text=None):
    if df.empty:
        raise RuntimeError(f"No strings to plot for {out_path}")

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.scatter(
        df["x"],
        df["y"],
        color="black",
        s=15,
    )

    for _, r in df.iterrows():
        ax.text(
            float(r["x"]),
            float(r["y"]),
            str(int(r["string"])),
            fontsize=6,
        )

    ax.set_title(title)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal", adjustable="box")

    xmin, xmax = df["x"].min(), df["x"].max()
    ymin, ymax = df["y"].min(), df["y"].max()

    xr = xmax - xmin
    yr = ymax - ymin
    pad = 0.03

    if xr == 0:
        xr = 1.0
    if yr == 0:
        yr = 1.0

    ax.set_xlim(xmin - pad * xr, xmax + pad * xr)
    ax.set_ylim(ymin - pad * yr, ymax + pad * yr)

    if count_text is not None:
        ax.text(
            0.02,
            0.98,
            count_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print("Saved plot:", out_path)


# ============================================================
# 3) READ GEOMETRY
# ============================================================

f = dataio.I3File(gcd_path, "r")

geo_frame = None
while f.more():
    fr = f.pop_frame()
    if fr.Stop == fr.Geometry:
        geo_frame = fr
        break

f.close()

if geo_frame is None:
    raise RuntimeError("Geometry frame not found in GCD file.")

geometry = geo_frame["I3Geometry"]
om_map = geometry.omgeo

print("Loaded I3OMGeoMap size:", len(om_map))


# ============================================================
# 4) COLLECT STRING XY
# ============================================================

xy_by_string = defaultdict(list)

for omkey, omgeo in om_map.items():
    s = int(omkey.string)
    pos = omgeo.position
    xy_by_string[s].append((float(pos.x), float(pos.y)))

rows = []
for s, xys in xy_by_string.items():
    rows.append(
        {
            "string": s,
            "x": xys[0][0],
            "y": xys[0][1],
        }
    )

string_xy_df = (
    pd.DataFrame(rows)
    .sort_values("string")
    .reset_index(drop=True)
)

print("Number of strings:", len(string_xy_df))


# ============================================================
# 5) READ SELECTED STRINGS FROM SUB-GEOMETRY CSV
# ============================================================

selected_ids = read_string_ids(csv_file)

print("Using selection file:", csv_file)
print("Number of selected strings:", len(selected_ids))


# ============================================================
# 6) FILTER SUB-GEOMETRY
# ============================================================

plot_df = string_xy_df[string_xy_df["string"].isin(selected_ids)].copy()

missing_ids = sorted(set(selected_ids) - set(plot_df["string"].tolist()))
if missing_ids:
    print("Warning: these string IDs were not found in the geometry:")
    print(missing_ids)

if plot_df.empty:
    raise RuntimeError("No matching strings found to plot.")


# ============================================================
# 7) MAKE SUB-GEOMETRY PLOT
# ============================================================

n_selected = len(plot_df)

plot_strings(
    df=plot_df,
    title=f"String Layout ({geometry_name})",
    out_path=out_png_sub,
    count_text=f"String count: {n_selected}",
)