import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse

# Same figure as before, with ONLY the thin white line issue fixed.
# The fix is: horizontal water shading stops before the seafloor band.

W, H = 1850, 820

fig = plt.figure(figsize=(14.5, 6.5), dpi=300)
ax = plt.axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")

def rect(x, y, w, h, color, ec=None, lw=1):
    face = "none" if color == "none" else color
    ax.add_patch(
        Rectangle(
            (x, y), w, h,
            facecolor=face,
            edgecolor=ec,
            linewidth=lw if ec else 0
        )
    )

def line(x1, y1, x2, y2, color="#bbbbbb", lw=1):
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, solid_capstyle="butt")

def txt(x, y, s, size=11, color="#222222", weight="normal", ha="left", va="center"):
    ax.text(
        x, y, s,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        family="serif"
    )

def circ(x, y, r, color, ec=None, lw=1):
    ax.add_patch(
        Circle(
            (x, y), r,
            facecolor=color,
            edgecolor=ec,
            linewidth=lw if ec else 0
        )
    )

# ---------- Left panel ----------
x_layer = 55
w_layer = 250
x_range = 340
x_density = 645

txt(x_layer, 70, "Layer", size=14, color="#333333", weight="bold", va="bottom")
txt(x_range, 70, "Radius range [m]", size=14, color="#333333", weight="bold", va="bottom")
txt(x_density, 70, r"Density [g/cm$^3$]", size=14, color="#333333", weight="bold", va="bottom")

layers = [
    ("#dceaf7", "Atmosphere",              "6,378,134 – 6,478,000", "0.000811", "#234a6f"),
    ("#9ec0dc", "Ocean water",             "6,375,477 – 6,378,134", "0.997",    "#18344f"),
    ("#cfc9bf", "Bedrock",                 "6,356,000 – 6,375,477", "2.65",     "#35322f"),
    ("#b8b1a8", "Inner crust",             "6,346,600 – 6,356,000", "2.90",     "#35322f"),
    ("#938f88", "Moho boundary",           "6,151,000 – 6,346,600", "~2.69",    "#ffffff"),
    ("#e3bf7b", "Upper transition zone",   "5,971,000 – 6,151,000", "~7.1",     "#4a3312"),
    ("#d29f58", "Middle transition zone",  "5,771,000 – 5,971,000", "~11.2",    "#4a3312"),
    ("#c78643", "Lower transition zone",   "5,701,000 – 5,771,000", "~5.3",     "#4a3312"),
    ("#b66f3d", "Lower mantle",            "3,480,000 – 5,701,000", "~7.9",     "#fff7f0"),
    ("#b25c5c", "Outer core",              "1,221,500 – 3,480,000", "~12.5",    "#fff7f7"),
    ("#7d2f2f", "Inner core",              "0 – 1,221,500",         "~13.1",    "#fff7f7"),
]

y = 92
ROW_H = 44
for fill, name, rr, density, tc in layers:
    rect(x_layer, y, w_layer, ROW_H, fill, ec="#d9d9d9", lw=0.8)
    txt(x_layer + 14, y + ROW_H / 2, name, size=11.5, color=tc)
    txt(x_range, y + ROW_H / 2, rr, size=11.5, color="#333333")
    txt(x_density, y + ROW_H / 2, density, size=11.5, color="#333333")
    y += ROW_H + 4

txt(x_range, y + 12, "Schematic (not to scale)", size=10, color="#888888")

# zoom box and guide lines
rect(x_layer, 140, w_layer, 44, "none", ec="#6f97b8", lw=1.5)
RX = 860
line(x_layer + w_layer, 144, RX, 108, color="#97b4cb", lw=0.9)
line(x_layer + w_layer, 184, RX, 720, color="#97b4cb", lw=0.9)

# ---------- Right panel ----------
OCEAN_H = 610
OCEAN_Y = 108
OCEAN_DEPTH = 2660

def depth_to_y(d):
    return OCEAN_Y + (d / OCEAN_DEPTH) * OCEAN_H

Y_WATER_ROCK = depth_to_y(2660)

# frame
rect(RX, OCEAN_Y, 560, OCEAN_H + 60, "#ffffff", ec="#d9d9d9", lw=0.9)

# sea surface
rect(RX, OCEAN_Y, 560, 18, "#dceaf7")
txt(RX + 280, OCEAN_Y + 9, "Sea surface", size=12.5, color="#234a6f", weight="bold", ha="center")

# water column
rect(RX, OCEAN_Y + 18, 560, OCEAN_H - 18, "#eef5fb")

# FIX: stop shading lines before the seafloor band
for yy in range(int(OCEAN_Y + 18), int(Y_WATER_ROCK - 24), 10):
    line(RX, yy, RX + 560, yy, color="#d9e8f4", lw=0.6)

# seafloor band
rect(RX, Y_WATER_ROCK - 24, 560, 24, "#7b7772")
txt(RX + 280, Y_WATER_ROCK - 12, "Seafloor (~2660 m)", size=12, color="#ffffff", weight="bold", ha="center")

# ruler
RULER_X = RX + 585
line(RULER_X, OCEAN_Y, RULER_X, Y_WATER_ROCK, color="#b9b9b9", lw=0.9)
txt(RULER_X + 10, OCEAN_Y, "0 m", size=10.5, color="#777777", va="center")

for depth, label, color, bold in [
    (500,  "500 m",                   "#777777", False),
    (1000, "1000 m",                  "#777777", False),
    (1500, "1500 m",                  "#777777", False),
    (2000, "2000 m",                  "#777777", False),
    (2100, "2100 m (detector depth)", "#8f2c4d", True),
    (2600, "2600 m",                  "#777777", False),
]:
    yy = depth_to_y(depth)
    line(RULER_X - 4, yy, RULER_X + 4, yy, color=color, lw=1.6 if bold else 0.9)
    txt(RULER_X + 10, yy, label, size=10.5, color=color, weight="bold" if bold else "normal")

# detector string
YBOT_STR = depth_to_y(2660)
YTOP_STR = depth_to_y(1660)
SX = RX + 300

STRING_COLOR = "#2e6f57"
DET_COLOR = "#8f2c4d"

line(SX, YTOP_STR, SX, YBOT_STR, color=STRING_COLOR, lw=2.2)
ax.add_patch(Ellipse((SX, YTOP_STR - 3), 26, 14, facecolor=STRING_COLOR, edgecolor="none"))
rect(SX - 10, YBOT_STR - 6, 20, 12, "#4a4a4a")

N_OM = 20
for i in range(N_OM):
    oy = YTOP_STR + 18 + i * ((YBOT_STR - YTOP_STR - 25) / (N_OM - 1))
    circ(SX, oy, 4.2, STRING_COLOR)

YCEN = depth_to_y(2100)
circ(SX, YCEN, 9.5, DET_COLOR, ec="#ffffff", lw=0.8)
line(SX + 10, YCEN, RULER_X - 5, YCEN, color=DET_COLOR, lw=1)

# "P-ONE string" label intentionally removed
line(SX + 26, YTOP_STR, SX + 26, YBOT_STR, color=STRING_COLOR, lw=0.8)
line(SX + 22, YTOP_STR, SX + 30, YTOP_STR, color=STRING_COLOR, lw=0.8)
line(SX + 22, YBOT_STR, SX + 30, YBOT_STR, color=STRING_COLOR, lw=0.8)
txt(SX + 36, (YTOP_STR + YBOT_STR) / 2, "~1000 m", size=10.2, color="#224f3f")

# medium labels
txt(RX + 18, OCEAN_Y + 48, "Ocean water", size=12, color="#325b7b", weight="bold")
txt(RX + 18, OCEAN_Y + 68, r"$\rho = 0.997\ \mathrm{g\,cm^{-3}}$", size=10.8, color="#46657d")

txt(RX + 18, Y_WATER_ROCK + 20, "Bedrock", size=12, color="#3a3734", weight="bold")
txt(RX + 18, Y_WATER_ROCK + 40, r"$\rho = 2.65\ \mathrm{g\,cm^{-3}}$", size=10.8, color="#4e4a46")

png_out = "earth_model.png"
pdf_out = "earth_model.pdf"

fig.savefig(png_out, dpi=1200, bbox_inches="tight", pad_inches=0.03)
fig.savefig(pdf_out, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

print(png_out)
print(pdf_out)