import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # <-- terminal/no-display için şart
import matplotlib.pyplot as plt
from icecube import dataio
from collections import defaultdict

# -----------------------
# 0) Input / Output
# -----------------------
gcd_path = "/project/6008051/pone_simulation/GCD_Library/PONE_800mGrid.i3.gz"

out_csv = "string_xy.csv"
out_png = "string_xy.png"
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

    x_ok = dx <= TOL
    y_ok = dy <= TOL

    if not (x_ok and y_ok):
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
# 3) Plot + save PNG
# -----------------------
fig = plt.figure(figsize=(14, 10))
ax = plt.gca()

ax.scatter(string_xy_df["x"], string_xy_df["y"], s=10)

# label all strings (çok kalabalık olabilir ama sen ss'te öyleydi)
for _, r in string_xy_df.iterrows():
    ax.text(r["x"], r["y"], str(int(r["string"])), fontsize=6)

ax.set_title("String XY Scatter Plot (from GCD)")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.savefig(out_png, dpi=200)
print("Saved plot:", out_png)
