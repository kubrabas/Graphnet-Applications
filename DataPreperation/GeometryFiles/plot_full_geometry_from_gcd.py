import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from icecube import dataio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the full string layout from a P-ONE GCD file."
    )
    parser.add_argument(
        "--gcd",
        default="/project/6008051/pone_simulation/GCD_Library/PONE_800mGrid_40mSpacing_40OMstring.i3.gz",
        help="Path to the .i3 or .i3.gz GCD file",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output filename stem, e.g. '340strings' → saves as 340strings.png (default: <gcd_basename>_full_geometry)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="Output image DPI (default: 200)"
    )
    return parser.parse_args()


def read_geometry_frame(gcd_path):
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

    return geo_frame


def collect_string_positions(om_map):
    xy_by_string = defaultdict(list)
    for omkey, omgeo in om_map.items():
        s = int(omkey.string)
        pos = omgeo.position
        xy_by_string[s].append((float(pos.x), float(pos.y)))

    rows = sorted(xy_by_string.items())
    strings = [s for s, _ in rows]
    xs = [xys[0][0] for _, xys in rows]
    ys = [xys[0][1] for _, xys in rows]
    return strings, xs, ys


def plot_geometry(strings, xs, ys, title, out_path, dpi):
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.scatter(xs, ys, color="black", s=15, zorder=2)

    for s, x, y in zip(strings, xs, ys):
        ax.text(x, y, str(s), fontsize=5, ha="left", va="bottom")

    ax.set_title(title)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal", adjustable="box")

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xr = max(xmax - xmin, 1.0)
    yr = max(ymax - ymin, 1.0)
    pad = 0.03
    ax.set_xlim(xmin - pad * xr, xmax + pad * xr)
    ax.set_ylim(ymin - pad * yr, ymax + pad * yr)

    ax.text(
        0.02, 0.98,
        f"String count: {len(strings)}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    args = parse_args()

    gcd_path = os.path.abspath(args.gcd)
    if not os.path.exists(gcd_path):
        raise FileNotFoundError(f"GCD file not found: {gcd_path}")

    out_dir = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/GeometryFiles"
    os.makedirs(out_dir, exist_ok=True)

    if args.name is not None:
        stem = args.name if args.name.endswith(".png") else args.name + ".png"
    else:
        stem = os.path.basename(gcd_path).split(".")[0] + "_full_geometry.png"
    out_path = os.path.join(out_dir, stem)

    print(f"Reading GCD: {gcd_path}")
    geo_frame = read_geometry_frame(gcd_path)
    om_map = geo_frame["I3Geometry"].omgeo
    print(f"OMGeoMap size: {len(om_map)}")

    strings, xs, ys = collect_string_positions(om_map)
    print(f"Total strings: {len(strings)}")

    gcd_basename = os.path.basename(gcd_path)
    plot_geometry(
        strings, xs, ys,
        title=f"Full String Layout - {gcd_basename}",
        out_path=out_path,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()


# =============================================================================
# Usage
# =============================================================================
# Basic (uses hardcoded GCD, saves as <gcd_basename>_full_geometry.png):
#   python3 /project/def-nahee/kbas/Graphnet-Applications/DataPreperation/GeometryFiles/plot_full_geometry_from_gcd.py
#
# Custom output name (saves as /project/def-nahee/kbas/Graphnet-Applications/Metadata/GeometryFiles/340strings.png):
#   python3 /project/def-nahee/kbas/Graphnet-Applications/DataPreperation/GeometryFiles/plot_full_geometry_from_gcd.py --name 340strings
#
# Override GCD path:
#   python3 /project/def-nahee/kbas/Graphnet-Applications/DataPreperation/GeometryFiles/plot_full_geometry_from_gcd.py --gcd /project/6008051/pone_simulation/GCD_Library/PONE_800mGrid.i3.gz --name 340_string_mc
#
# Custom DPI:
#   python3 /project/def-nahee/kbas/Graphnet-Applications/DataPreperation/GeometryFiles/plot_full_geometry_from_gcd.py --name 340strings --dpi 300
