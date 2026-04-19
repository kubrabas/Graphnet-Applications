"""
P-ONE Simulation Pipeline — PDF generator
==========================================
Requires: cairosvg  (pip install cairosvg)

Run with:
    python clsim_path.py

Output: p-one_pipeline.pdf  (same folder)

LaTeX usage:
    \\includegraphics[width=\\textwidth]{figures/p-one_pipeline}

CUSTOMISATION
-------------
All layout constants are at the top of each section.
Colors are defined in the COLORS dict.
"""

# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    # LeptonInjector (purple ramp)
    "li_fill":   "#EEEDFE",
    "li_stroke": "#534AB7",
    "li_title":  "#3C3489",
    "li_sub":    "#534AB7",

    # PROPOSAL (amber ramp)
    "pr_fill":   "#FAEEDA",
    "pr_stroke": "#BA7517",
    "pr_title":  "#633806",

    # Energy-loss processes (coral ramp)
    "ep_fill":   "#FAECE7",
    "ep_stroke": "#993C1D",
    "ep_title":  "#712B13",

    # clsim / PPC (teal ramp)
    "cl_fill":   "#E1F5EE",
    "cl_stroke": "#0F6E56",
    "cl_title":  "#085041",
    "cl_sub":    "#0F6E56",

    # Connector arrows
    "edge":      "#888780",
}

FONT = "Arial, Helvetica, sans-serif"

# ── Canvas ─────────────────────────────────────────────────────────────────────
W, H = 680, 504


# ── SVG helpers ───────────────────────────────────────────────────────────────

def rect(x, y, w, h, fill, stroke, rx=8, sw=0.75):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def text(content, x, y, color, size=14, weight=500, anchor="middle", baseline="central"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="{baseline}" '
            f'font-family="{FONT}" font-size="{size}px" font-weight="{weight}" '
            f'fill="{color}">{content}</text>')

def line(x1, y1, x2, y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{COLORS["edge"]}" stroke-width="1.2" '
            f'marker-end="url(#arr)" fill="none"/>')

def path(d):
    return (f'<path d="{d}" stroke="{COLORS["edge"]}" stroke-width="1.2" '
            f'fill="none" marker-end="url(#arr)"/>')

def legend_item(x, y, fill, stroke, label, label_color):
    r = rect(x, y, 12, 12, fill, stroke, rx=3, sw=0.75)
    t = text(label, x + 18, y + 6, label_color, size=12, weight=400, anchor="start")
    return r + "\n  " + t


# ── Build SVG ─────────────────────────────────────────────────────────────────

def build_svg():
    parts = []

    # ── Header & defs ────────────────────────────────────────────────────────
    parts.append(f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="{COLORS["edge"]}"
          stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>''')

    # ── ROW 1 — νμ / ν̄μ  LeptonInjector ────────────────────────────────────
    # Box: x=299 y=30 w=220 h=56   cx=409
    r1x, r1y, r1w, r1h, r1cx = 299, 30, 220, 56, 409
    parts.append(f"\n  <!-- ROW 1: LeptonInjector -->")
    parts.append("  " + rect(r1x, r1y, r1w, r1h,
                              COLORS["li_fill"], COLORS["li_stroke"]))
    parts.append("  " + text("&#957;&#956; / &#957;&#772;&#956;",
                               r1cx, r1y + 18, COLORS["li_title"], size=14, weight=500))
    parts.append("  " + text("LeptonInjector",
                               r1cx, r1y + 38, COLORS["li_sub"], size=12, weight=400))

    # Root branch: split at y=110
    parts.append(f"\n  <!-- Root branch arrows -->")
    parts.append("  " + path(f"M{r1cx} {r1y+r1h} L{r1cx} 110 L255 110 L255 134"))
    parts.append("  " + path(f"M{r1cx} {r1y+r1h} L{r1cx} 110 L563 110 L563 134"))

    # ── ROW 2 — μ⁻/μ⁺  and  Hadrons ─────────────────────────────────────────
    parts.append(f"\n  <!-- ROW 2: mu and hadrons -->")

    # μ⁻ / μ⁺
    parts.append("  " + rect(170, 134, 170, 44,
                               COLORS["li_fill"], COLORS["li_stroke"]))
    parts.append("  " + text("&#956;&#8315; / &#956;&#8314;",
                               255, 156, COLORS["li_title"]))

    # Hadrons
    parts.append("  " + rect(498, 134, 130, 44,
                               COLORS["li_fill"], COLORS["li_stroke"]))
    parts.append("  " + text("Hadrons", 563, 156, COLORS["li_title"]))

    # μ → PROPOSAL
    parts.append("  " + line(255, 178, 255, 226))

    # Hadrons long arrow → clsim (bypasses PROPOSAL)
    parts.append("  " + line(563, 178, 563, 408))

    # ── ROW 3 — PROPOSAL ────────────────────────────────────────────────────
    parts.append(f"\n  <!-- ROW 3: PROPOSAL -->")
    parts.append("  " + rect(175, 226, 160, 44,
                               COLORS["pr_fill"], COLORS["pr_stroke"]))
    parts.append("  " + text("PROPOSAL", 255, 248, COLORS["pr_title"]))

    # Fan from PROPOSAL bottom (255, 270) → 5 energy-loss boxes (y=316)
    # Horizontal junction at y=292
    fan_targets = [65, 160, 255, 350, 445]   # cx of each process box
    parts.append(f"\n  <!-- Fan arrows: PROPOSAL → energy-loss processes -->")
    for cx in fan_targets:
        if cx == 255:
            parts.append("  " + line(255, 270, 255, 316))
        else:
            parts.append("  " + path(f"M255 270 L255 292 L{cx} 292 L{cx} 316"))

    # ── ROW 4 — Energy-loss process types ───────────────────────────────────
    # w=85, y=316, h=44; x values: 23,118,213,308,403
    process_boxes = [
        (23,  65,  "&#956;&#177; track"),   # μ± track segments
        (118, 160, "Brems"),                # Bremsstrahlung
        (213, 255, "&#916;E"),              # Ionization / delta-E
        (308, 350, "PairProd"),             # Pair production
        (403, 445, "NuclInt"),              # Nuclear interaction
    ]
    parts.append(f"\n  <!-- ROW 4: energy-loss process types -->")
    for (bx, cx, label) in process_boxes:
        parts.append("  " + rect(bx, 316, 85, 44,
                                   COLORS["ep_fill"], COLORS["ep_stroke"]))
        parts.append("  " + text(label, cx, 338, COLORS["ep_title"]))

    # Process → clsim arrows
    parts.append(f"\n  <!-- Arrows: process types → clsim boxes -->")
    for cx in fan_targets:
        parts.append("  " + line(cx, 360, cx, 408))

    # ── ROW 5 — clsim / PPC ─────────────────────────────────────────────────
    # Same cx as Row 4; y=408 h=56 w=85
    clsim_boxes = [
        (23,  65,  "clsim", "muon-like"),
        (118, 160, "clsim", "EM cascade"),
        (213, 255, "clsim", "EM cascade"),
        (308, 350, "clsim", "EM cascade"),
        (403, 445, "clsim", "hadronic"),
    ]
    parts.append(f"\n  <!-- ROW 5: clsim / PPC boxes -->")
    for (bx, cx, title, subtitle) in clsim_boxes:
        parts.append("  " + rect(bx, 408, 85, 56,
                                   COLORS["cl_fill"], COLORS["cl_stroke"]))
        parts.append("  " + text(title,    cx, 426, COLORS["cl_title"], size=14, weight=500))
        parts.append("  " + text(subtitle, cx, 444, COLORS["cl_sub"],   size=12, weight=400))

    # Hadrons direct clsim box: x=498 w=130 cx=563
    parts.append("  " + rect(498, 408, 130, 56,
                               COLORS["cl_fill"], COLORS["cl_stroke"]))
    parts.append("  " + text("clsim",          563, 426, COLORS["cl_title"], size=14, weight=500))
    parts.append("  " + text("hadronic casc.", 563, 444, COLORS["cl_sub"],   size=12, weight=400))

    # ── Legend ────────────────────────────────────────────────────────────────
    parts.append(f"\n  <!-- Legend -->")
    legend = [
        (30,  COLORS["li_fill"], COLORS["li_stroke"], "LeptonInjector",       COLORS["li_title"]),
        (165, COLORS["pr_fill"], COLORS["pr_stroke"], "PROPOSAL",              COLORS["pr_title"]),
        (265, COLORS["ep_fill"], COLORS["ep_stroke"], "Energy-loss processes", COLORS["ep_title"]),
        (445, COLORS["cl_fill"], COLORS["cl_stroke"], "clsim / PPC",           COLORS["cl_title"]),
    ]
    for (x, fill, stroke, label, lc) in legend:
        parts.append("  " + legend_item(x, 482, fill, stroke, label, lc))

    parts.append("\n</svg>")
    return "\n".join(parts)


# ── Write file ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cairosvg

    output_path = "mc_pipeline.pdf"
    svg_content = build_svg()

    cairosvg.svg2pdf(bytestring=svg_content.encode("utf-8"), write_to=output_path)

    print(f"Saved: {output_path}")