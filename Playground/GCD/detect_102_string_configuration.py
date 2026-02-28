#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # terminal/no-display için şart
import matplotlib.pyplot as plt

from icecube import dataio
from collections import defaultdict

# -----------------------
# 0) Input / Output
# -----------------------
gcd_path = "/project/6008051/pone_simulation/GCD_Library/PONE_800mGrid.i3.gz"

out_csv = "string_xy.csv"
out_png = "string_xy.png"
out_png_lines = "string_xy_with_lines.png"
out_bad_csv = "strings_with_xy_variation.csv"

# -----------------------
# 1) Read Geometry frame
# -----------------------
f = dataio.I3File(gcd_path, "r")

geo_frame = None
while f.more():
    fr = f.pop_frame()
    if fr.Stop == fr.Geometry:
        geo_frame = fr
        break

if geo_frame is None:
    raise RuntimeError("Geometry frame not found in GCD file.")

geometry = geo_frame["I3Geometry"]
om_map = geometry.omgeo  # I3OMGeoMap

print("Loaded I3OMGeoMap size:", len(om_map))

# -----------------------
# 2) Collect x,y per string and check constancy
# -----------------------
xy_by_string = defaultdict(list)

for omkey, omgeo in om_map.items():
    s = int(omkey.string)
    pos = omgeo.position
    xy_by_string[s].append((float(pos.x), float(pos.y)))

TOL = 1e-6  # meters
bad_strings = []
rows = []

for s, xys in xy_by_string.items():
    xs = np.array([p[0] for p in xys], dtype=float)
    ys = np.array([p[1] for p in xys], dtype=float)

    dx = float(xs.max() - xs.min())
    dy = float(ys.max() - ys.min())

    if (dx > TOL) or (dy > TOL):
        bad_strings.append(
            {
                "string": s,
                "x_min": float(xs.min()),
                "x_max": float(xs.max()),
                "y_min": float(ys.min()),
                "y_max": float(ys.max()),
                "dx": dx,
                "dy": dy,
                "n_points": int(len(xys)),
            }
        )

    # representative x,y: first OM's coords
    rows.append({"string": s, "x": float(xs[0]), "y": float(ys[0])})

string_xy_df = pd.DataFrame(rows).sort_values("string").reset_index(drop=True)

print("Number of strings:", len(string_xy_df))
print("Strings with non-constant x/y across OMs:", len(bad_strings))

# Save string table
string_xy_df.to_csv(out_csv, index=False)
print("Wrote:", out_csv)

# Save bad strings table (if any)
if bad_strings:
    bad_df = pd.DataFrame(bad_strings).sort_values(["dx", "dy"], ascending=False)
    bad_df.to_csv(out_bad_csv, index=False)
    print("⚠ Wrote:", out_bad_csv)
else:
    print("✅ All strings have constant x and y (within tolerance).")

# -----------------------
# 3) Helper: draw extended line through two string IDs
# -----------------------
def add_extended_line_by_strings(ax, string_xy_df, s1, s2, pad_frac=0.02, **plot_kwargs):
    """
    Two string ID -> draw the infinite line passing through their (x,y),
    extended to cover the whole plotted array (axis limits).
    """
    s1 = int(s1); s2 = int(s2)

    r1 = string_xy_df.loc[string_xy_df["string"] == s1]
    r2 = string_xy_df.loc[string_xy_df["string"] == s2]

    if r1.empty or r2.empty:
        missing = []
        if r1.empty: missing.append(str(s1))
        if r2.empty: missing.append(str(s2))
        print(f"⚠ WARNING: String id not found, skipping line: {s1}-{s2} (missing: {', '.join(missing)})")
        return

    x1, y1 = float(r1["x"].iloc[0]), float(r1["y"].iloc[0])
    x2, y2 = float(r2["x"].iloc[0]), float(r2["y"].iloc[0])

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    xr = xmax - xmin
    yr = ymax - ymin
    xmin_p = xmin - pad_frac * xr
    xmax_p = xmax + pad_frac * xr
    ymin_p = ymin - pad_frac * yr
    ymax_p = ymax + pad_frac * yr

    # Vertical line
    if abs(x2 - x1) < 1e-12:
        ax.plot([x1, x1], [ymin_p, ymax_p], **plot_kwargs)
        return

    # y = m x + b
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    y_at_xmin = m * xmin_p + b
    y_at_xmax = m * xmax_p + b

    ax.plot([xmin_p, xmax_p], [y_at_xmin, y_at_xmax], **plot_kwargs)

# -----------------------
# 4) Plot base scatter + labels
# -----------------------
fig = plt.figure(figsize=(14, 10))
ax = plt.gca()

# Base points (all strings): blue circles
ax.scatter(string_xy_df["x"], string_xy_df["y"], s=10, marker="o")  # default blue

# label all strings
for _, r in string_xy_df.iterrows():
    ax.text(float(r["x"]), float(r["y"]), str(int(r["string"])), fontsize=6)

ax.set_title("String XY Scatter Plot (from GCD)")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.set_aspect("equal", adjustable="box")

# Axis limits: data range + padding
xmin, xmax = float(string_xy_df["x"].min()), float(string_xy_df["x"].max())
ymin, ymax = float(string_xy_df["y"].min()), float(string_xy_df["y"].max())
xr = xmax - xmin
yr = ymax - ymin
pad = 0.03
ax.set_xlim(xmin - pad * xr, xmax + pad * xr)
ax.set_ylim(ymin - pad * yr, ymax + pad * yr)

plt.tight_layout()
plt.savefig(out_png, dpi=200)
print("Saved base plot:", out_png)

# -----------------------
# 5) Highlight selected strings (SINGLE GROUP)
# -----------------------
highlight_ids = sorted(set([
    200, 201, 117, 118, 114, 115, 95, 96, 97, 98, 99, 100, 77, 78, 79, 80, 81, 61, 62, 44, 45, 46, 29, 30,
    180, 181, 182,
    161, 162,
    121, 122, 124, 125,
    102, 103, 104, 105, 106, 107,
    84, 85, 86, 87, 88,
    68, 69, 117, 118, 114, 115, 95, 96, 97, 98, 99, 100, 77, 78, 79, 80, 81, 61, 62, 44, 45, 46, 29, 30, 
    51, 52, 53, 153, 154, 172, 173, 174, 176, 177, 192, 193, 196, 197, 214, 215, 216, 217, 232, 233, 234, 235, 236,
    36, 37,  158, 157, 178, 218, 237, 250, 251, 253, 254, 165, 166, 184, 185, 186, 204, 205, 222, 223, 224, 225, 226 , 240 , 241, 242, 243, 244, 245, 258, 259, 261, 262
]))

sub = string_xy_df[string_xy_df["string"].isin(highlight_ids)]
missing = sorted(set(highlight_ids) - set(sub["string"].tolist()))
if missing:
    print(f"⚠ WARNING: missing highlight string IDs: {missing}")

# One style for all highlighted points
if not sub.empty:
    ax.scatter(
        sub["x"], sub["y"],
        color="red",
        marker="*",
        s=120,
        edgecolors="k",
        linewidths=0.6,
        zorder=6,
        label="highlighted strings",
    )

# -----------------------
# 6) Add requested lines (string pairs)
# -----------------------
pairs = [
    (157, 176),
    (158, 178),
    (180, 200),
    (201, 182),
    (184, 165),
    (174, 155),
    (161, 162),
    (124, 125),
]

for a, b in pairs:
    add_extended_line_by_strings(
        ax,
        string_xy_df,
        a, b,
        color="red",
        lw=2,
        alpha=0.85,
        zorder=4,
    )

# Optional legend
ax.legend(loc="best", fontsize=10)

# Save final plot with highlights + lines
plt.tight_layout()
plt.savefig(out_png_lines, dpi=200)
print("Saved plot with highlights + lines:", out_png_lines)
