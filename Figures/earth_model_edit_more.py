from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1600, 760
img = Image.new('RGB', (W, H), '#ffffff')
draw = ImageDraw.Draw(img)

# Mac font paths
FONT_PATHS = [
    '/System/Library/Fonts/Helvetica.ttc',
    '/Library/Fonts/Arial.ttf',
    '/System/Library/Fonts/Arial.ttf',
]

def get_font(size=11, bold=False):
    for path in FONT_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    return ImageFont.load_default()

def rect_fill(x, y, w, h, hex_color, alpha=255):
    r, g, b = int(hex_color[1:3],16), int(hex_color[3:5],16), int(hex_color[5:7],16)
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([x, y, x+w, y+h], fill=(r, g, b, alpha))
    composed = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    img.paste(composed)

def text(x, y, s, size=11, color='#222222', bold=False, anchor='lt'):
    fnt = get_font(size, bold)
    draw.text((x, y), s, fill=color, font=fnt, anchor=anchor)

def hline(x1, y, x2, color='#cccccc', w=1):
    draw.line([(x1, y), (x2, y)], fill=color, width=w)

def vline(x, y1, y2, color='#cccccc', w=1):
    draw.line([(x, y1), (x, y2)], fill=color, width=w)

def circle(x, y, r, color):
    draw.ellipse([x-r, y-r, x+r, y+r], fill=color)


# ── LEFT PANEL ──
text(40, 75, 'Layer', 12, '#444444', bold=True)
text(240, 75, 'Radius range [m]', 12, '#444444', bold=True)
text(470, 75, 'Density [g/cm3]', 12, '#444444', bold=True)

layers = [
    ('#B5D4F4', 190, 'atmosphere',        '6,378,134 - 6,478,000', '0.000811', '#0C447C'),
    ('#3377BB', 220, 'ocean water',        '6,375,477 - 6,378,134', '0.997',    '#ffffff'),
    ('#C8C6BC', 220, 'bedrock',            '6,356,000 - 6,375,477', '2.65',     '#2C2C2A'),
    ('#B0AEA5', 220, 'inner crust',        '6,346,600 - 6,356,000', '2.90',     '#2C2C2A'),
    ('#888780', 220, 'moho boundary',      '6,151,000 - 6,346,600', '~2.69',    '#2C2C2A'),
    ('#EFA020', 200, 'upper transition',   '5,971,000 - 6,151,000', '~7.1',     '#412402'),
    ('#E88800', 210, 'middle transition',  '5,771,000 - 5,971,000', '~11.2',    '#412402'),
    ('#DF7800', 220, 'lower transition',   '5,701,000 - 5,771,000', '~5.3',     '#412402'),
    ('#C86800', 230, 'lower mantle',       '3,480,000 - 5,701,000', '~7.9',     '#412402'),
    ('#C04040', 210, 'outer core',         '1,221,500 - 3,480,000', '~12.5',    '#FCEBEB'),
    ('#801818', 230, 'inner core',         '0 - 1,221,500',         '~13.1',    '#FCEBEB'),
]

y = 92
ROW_H = 42
for hex_c, alpha, name, rrange, density, tc in layers:
    rect_fill(40, y, 185, ROW_H, hex_c, alpha)
    draw.rectangle([40, y, 224, y+ROW_H], outline='#dddddd', width=1)
    text(48, y + ROW_H//2 - 6, name, 12, tc)
    text(240, y + ROW_H//2 - 6, rrange, 12, '#333333')
    text(470, y + ROW_H//2 - 6, density, 12, '#333333')
    y += ROW_H + 2

text(240, y + 12, 'not to scale', 11, '#aaaaaa')

# zoom box
draw.rectangle([40, 134, 224, 176], outline='#3377BB', width=2)
RX = 650
draw.line([(225, 138), (RX, 105)], fill='#3377BB', width=1)
draw.line([(225, 176), (RX, 685)], fill='#3377BB', width=1)

# ── RIGHT PANEL ──
OCEAN_H = 580
OCEAN_Y = 105
OCEAN_DEPTH = 2660

def depth_to_y(d):
    return OCEAN_Y + int(d / OCEAN_DEPTH * OCEAN_H)

rect_fill(RX, OCEAN_Y, 520, 18, '#B5D4F4', 210)
text(RX + 260, OCEAN_Y + 4, 'sea surface  -  0 m', 12, '#0C447C', bold=True, anchor='mt')

rect_fill(RX, OCEAN_Y+18, 520, OCEAN_H-18, '#4488CC', 35)

Y_ICEAIR = depth_to_y(2660) - 35
rect_fill(RX, Y_ICEAIR, 520, 35, '#1E5FA0', 230)
text(RX+12, Y_ICEAIR+5,  'iceair_boundary', 13, '#ffffff', bold=True)
text(RX+12, Y_ICEAIR+21, 'r = 6,378,134 m  -  ocean water  -  0.997 g/cm3', 11, '#ddeeff')

Y_ROCKICE = Y_ICEAIR + 35
rect_fill(RX, Y_ROCKICE, 520, 35, '#C0BEBA', 240)
text(RX+12, Y_ROCKICE+5,  'rockice_boundary', 13, '#2C2C2A', bold=True)
text(RX+12, Y_ROCKICE+21, 'r = 6,375,477 m  -  bedrock  -  2.65 g/cm3', 11, '#333333')

Y_SEAFLOOR = Y_ROCKICE + 35
rect_fill(RX, Y_SEAFLOOR, 520, 22, '#888780', 210)
text(RX+260, Y_SEAFLOOR+5, 'seafloor  ~2660 m', 12, '#ffffff', bold=True, anchor='mt')

hline(RX, Y_ICEAIR,   RX+520, '#1E5FA0', 2)
hline(RX, Y_ROCKICE,  RX+520, '#888888', 2)
hline(RX, Y_SEAFLOOR, RX+520, '#444444', 2)

RULER_X = RX + 540
vline(RULER_X, OCEAN_Y, Y_SEAFLOOR, '#cccccc', 1)
text(RULER_X+6, OCEAN_Y, '0 m', 11, '#888888')

for depth, label, color, bld in [
    (500,  '500 m',                   '#888888', False),
    (1000, '1000 m',                  '#888888', False),
    (1500, '1500 m',                  '#888888', False),
    (2000, '2000 m',                  '#888888', False),
    (2100, '2100 m (DetectorDepth)',  '#CC4477', True),
    (2600, '2600 m',                  '#888888', False),
]:
    yy = depth_to_y(depth)
    hline(RULER_X-4, yy, RULER_X+4, color, 2 if bld else 1)
    text(RULER_X+6, yy-6, label, 11, color, bold=bld)

YBOT_STR = depth_to_y(2660)
YTOP_STR = depth_to_y(2660 - 1000)
SX = RX + 200

draw.line([(SX, YTOP_STR), (SX, YBOT_STR)], fill='#1D9E75', width=5)
draw.ellipse([SX-14, YTOP_STR-10, SX+14, YTOP_STR+5], fill='#1D9E75')
text(SX, YTOP_STR-18, 'float', 10, '#085041', anchor='mt')
draw.rectangle([SX-10, YBOT_STR-5, SX+10, YBOT_STR+8], fill='#444444')
text(SX, YBOT_STR+14, 'anchor', 10, '#444444', anchor='mt')

N_OM = 20
for i in range(N_OM):
    oy = YTOP_STR + 18 + i * int((YBOT_STR - YTOP_STR - 25) / (N_OM - 1))
    circle(SX, oy, 4, '#1D9E75')

YCEN = depth_to_y(2100)
circle(SX, YCEN, 11, '#CC4477')
draw.line([(SX+11, YCEN), (RULER_X-5, YCEN)], fill='#CC4477', width=1)

text(SX-80, (YTOP_STR+YBOT_STR)//2-10, 'P-ONE string', 13, '#085041', bold=True)
text(SX-80, (YTOP_STR+YBOT_STR)//2+8,  '~1000 m  -  20 OMs', 11, '#085041')

draw.line([(SX+22, YTOP_STR), (SX+22, YBOT_STR)], fill='#1D9E75', width=1)
hline(SX+18, YTOP_STR, SX+26, '#1D9E75', 1)
hline(SX+18, YBOT_STR, SX+26, '#1D9E75', 1)
text(SX+28, (YTOP_STR+YBOT_STR)//2-6, '~1000 m', 10, '#085041')

# img.save('earth_model.png', dpi=(200, 200))
img_big = img.resize((W * 3, H * 3), Image.LANCZOS)
img_big.save('earth_model.png', dpi=(300, 300))

