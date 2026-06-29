"""Rotating 3D event display helper for accepted pulse maps.

Examples
--------
Draw one event inline in a notebook.  The GCD path is resolved from
``Metadata/paths.py`` by passing the MC name:

    from pathlib import Path
    from icecube import dataio, LeptonInjector, simclasses
    from rotating_event_view import rotating_event_view

    tau_file = Path(
        "/home/kbas/scratch/String340MC_pone_offline_version3/"
        "Tau_PMT_Response/tau_gen_1212.i3.gz"
    )
    pulsemap_key = "Accepted_PulseMap_340String"

    def accepted_pulse_count(frame):
        if pulsemap_key not in frame:
            return 0
        pulse_map = frame[pulsemap_key]
        if not hasattr(pulse_map, "items"):
            return 0
        return sum(len(pulses) for _, pulses in pulse_map.items())

    i3_file = dataio.I3File(str(tau_file), "r")
    try:
        while i3_file.more():
            frame = i3_file.pop_frame()
            if accepted_pulse_count(frame) > 0:
                break
        else:
            raise RuntimeError(f"No non-empty {pulsemap_key} found in {tau_file}")
    finally:
        i3_file.close()

    fig, ani, saved_path = rotating_event_view(
        frame,
        mc="340StringMC",
        mmc_track=True,
        show_full_layout=True,
        save=False,
        pulsemap_key=pulsemap_key,
        title="tau_gen_1212 first accepted-pulse frame: rotating 3D view",
    )

Save the same view as a single-event GIF:

    rotating_event_view(
        frame,
        mc="340StringMC",
        mmc_track=True,
        show_full_layout=True,
        save=True,
        output_path="tau_gen_1212_single_event_rotating_3d.gif",
        pulsemap_key=pulsemap_key,
    )
"""

from collections import defaultdict
import importlib.util
from pathlib import Path

from matplotlib import animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt

from icecube import dataio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS_PY = PROJECT_ROOT / "Metadata" / "paths.py"
DEFAULT_MC = "340StringMC"
DEFAULT_PULSEMAP_KEY = "Accepted_PulseMap_340String"
TRACK_COLORS = ["crimson", "royalblue", "darkorange", "purple", "seagreen", "black"]


def load_gcd_paths():
    spec = importlib.util.spec_from_file_location("graphnet_application_paths", PATHS_PY)
    paths_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paths_module)
    return paths_module.GCD


def resolve_gcd_path(mc=DEFAULT_MC, gcd_path=None):
    if gcd_path is not None:
        return gcd_path

    gcd_paths = load_gcd_paths()
    if mc in gcd_paths:
        return gcd_paths[mc]

    normalized = str(mc).lower().replace("_", "").replace("-", "")
    aliases = {
        "340": "340StringMC",
        "340string": "340StringMC",
        "340stringmc": "340StringMC",
        "string340": "340StringMC",
        "string340mc": "340StringMC",
        "spring": "Spring2026MC",
        "spring2026": "Spring2026MC",
        "spring2026mc": "Spring2026MC",
    }
    key = aliases.get(normalized)
    if key is not None:
        return gcd_paths[key]

    valid = ", ".join(sorted(gcd_paths))
    raise ValueError(f"Unknown mc={mc!r}. Expected one of: {valid}")


def read_geometry_frame(gcd_path=None, mc=DEFAULT_MC):
    gcd_path = resolve_gcd_path(mc=mc, gcd_path=gcd_path)
    gcd_file = dataio.I3File(str(gcd_path), "r")
    try:
        while gcd_file.more():
            frame = gcd_file.pop_frame()
            if frame.Stop == frame.Geometry:
                return frame
    finally:
        gcd_file.close()
    raise RuntimeError(f"No Geometry frame found in {gcd_path}")


def pe_time(pe):
    if hasattr(pe, "time"):
        return float(pe.time)
    return float(pe.Time)


def frame_to_om_hits(frame, geometry, pulsemap_key=DEFAULT_PULSEMAP_KEY):
    if pulsemap_key not in frame:
        print(f"{pulsemap_key} not found in frame; no OM-level hits to plot.")
        return [], [], [], [], [], []

    pulse_map = frame[pulsemap_key]
    if not hasattr(pulse_map, "items"):
        raise TypeError(
            f"{pulsemap_key} is not a typed map. "
            "Restart the kernel and run this after importing simclasses."
        )

    if not pulse_map:
        print(f"{pulsemap_key} is empty; no OM-level hits to plot.")
        return [], [], [], [], [], []

    om_hits = defaultdict(lambda: {"x": [], "y": [], "z": [], "pulse_count": 0, "time": None})

    for omkey, pulses in pulse_map.items():
        if omkey not in geometry:
            continue

        pos = geometry[omkey].position
        om_id = (int(omkey.string), int(omkey.om))

        om_hits[om_id]["x"].append(float(pos.x))
        om_hits[om_id]["y"].append(float(pos.y))
        om_hits[om_id]["z"].append(float(pos.z))
        om_hits[om_id]["pulse_count"] += len(pulses)

        for pe in pulses:
            t = pe_time(pe)
            if om_hits[om_id]["time"] is None or t < om_hits[om_id]["time"]:
                om_hits[om_id]["time"] = t

    xs, ys, zs, pulse_counts, times = [], [], [], [], []
    for hit in om_hits.values():
        if hit["pulse_count"] <= 0 or hit["time"] is None:
            continue
        xs.append(sum(hit["x"]) / len(hit["x"]))
        ys.append(sum(hit["y"]) / len(hit["y"]))
        zs.append(sum(hit["z"]) / len(hit["z"]))
        pulse_counts.append(hit["pulse_count"])
        times.append(hit["time"])

    if not xs:
        print(f"No OM-level hits found in {pulsemap_key}; no OM-level hits to plot.")
        return [], [], [], [], [], []

    sizes = [18 + 8 * min(n, 20) for n in pulse_counts]
    return xs, ys, zs, pulse_counts, times, sizes


def geometry_to_om_layout(geometry):
    om_positions = defaultdict(lambda: {"x": [], "y": [], "z": []})
    for omkey, omgeo in geometry.items():
        om_id = (int(omkey.string), int(omkey.om))
        pos = omgeo.position
        om_positions[om_id]["x"].append(float(pos.x))
        om_positions[om_id]["y"].append(float(pos.y))
        om_positions[om_id]["z"].append(float(pos.z))

    xs, ys, zs = [], [], []
    for pos in om_positions.values():
        xs.append(sum(pos["x"]) / len(pos["x"]))
        ys.append(sum(pos["y"]) / len(pos["y"]))
        zs.append(sum(pos["z"]) / len(pos["z"]))
    return xs, ys, zs


def mmc_xi_xf_segments(frame):
    if "MMCTrackList" not in frame:
        return []

    segments = []
    for idx, track in enumerate(frame["MMCTrackList"]):
        try:
            xi = (float(track.xi), float(track.yi), float(track.zi))
            xf = (float(track.xf), float(track.yf), float(track.zf))
        except AttributeError:
            continue

        particle = getattr(track, "particle", None)
        particle_type = str(getattr(particle, "type", f"track_{idx}"))
        segments.append(
            {
                "xi": xi,
                "xf": xf,
                "color": TRACK_COLORS[idx % len(TRACK_COLORS)],
                "label": f"MMC {idx}: {particle_type}",
            }
        )
    return segments


def clean_3d_axes(ax):
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_alpha(0.0)
        axis.pane.set_edgecolor("white")
        axis._axinfo["grid"]["linewidth"] = 0
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.set_proj_type("ortho")


def common_equal_limits(all_xs, all_ys, all_zs, pad_fraction=0.08):
    xmid = 0.5 * (min(all_xs) + max(all_xs))
    ymid = 0.5 * (min(all_ys) + max(all_ys))
    zmid = 0.5 * (min(all_zs) + max(all_zs))
    span = max(max(all_xs) - min(all_xs), max(all_ys) - min(all_ys), max(all_zs) - min(all_zs))
    span = span * (1 + pad_fraction) or 1.0
    half = 0.5 * span
    return (xmid - half, xmid + half), (ymid - half, ymid + half), (zmid - half, zmid + half)


def evenly_spaced_ticks(limits, n_ticks=5):
    lo, hi = limits
    if n_ticks <= 1:
        return [0.5 * (lo + hi)]
    step = (hi - lo) / (n_ticks - 1)
    return [lo + i * step for i in range(n_ticks)]


def default_output_path(frame, output_path):
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "rotating_event_view.gif"
    else:
        output_path = Path(output_path)

    if output_path.suffix.lower() != ".gif":
        run_id = None
        event_id = None
        if "I3EventHeader" in frame:
            header = frame["I3EventHeader"]
            run_id = int(getattr(header, "run_id", 0))
            event_id = int(getattr(header, "event_id", 0))
        name = f"run{run_id:04d}_event{event_id:03d}_rotating_3d.gif" if run_id or event_id else "rotating_event_view.gif"
        output_path = output_path / name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def rotating_event_view(
    frame,
    mmc_track=False,
    save=True,
    output_path=None,
    geometry=None,
    gcd_path=None,
    mc=DEFAULT_MC,
    pulsemap_key=DEFAULT_PULSEMAP_KEY,
    title=None,
    fps=15,
    dpi=120,
    display_html=True,
    show_full_layout=False,
):
    if geometry is None:
        geometry = read_geometry_frame(gcd_path=gcd_path, mc=mc)["I3Geometry"].omgeo

    xs, ys, zs, pulse_counts, times, sizes = frame_to_om_hits(frame, geometry, pulsemap_key)
    segments = mmc_xi_xf_segments(frame) if mmc_track else []
    layout_xs, layout_ys, layout_zs = geometry_to_om_layout(geometry) if show_full_layout else ([], [], [])

    all_xs = list(xs) + list(layout_xs) + [coord for seg in segments for coord in (seg["xi"][0], seg["xf"][0])]
    all_ys = list(ys) + list(layout_ys) + [coord for seg in segments for coord in (seg["xi"][1], seg["xf"][1])]
    all_zs = list(zs) + list(layout_zs) + [coord for seg in segments for coord in (seg["xi"][2], seg["xf"][2])]
    if all_xs and all_ys and all_zs:
        xlim, ylim, zlim = common_equal_limits(all_xs, all_ys, all_zs)
    else:
        xlim, ylim, zlim = (-1, 1), (-1, 1), (-1, 1)

    if title is None:
        if "I3EventHeader" in frame:
            header = frame["I3EventHeader"]
            title = f"Run {int(header.run_id)} / Event {int(header.event_id)}"
        else:
            title = "Rotating 3D event view"

    fig = plt.figure(figsize=(9, 8))
    fig.subplots_adjust(left=0.03, right=0.88, bottom=0.04, top=0.92)
    ax = fig.add_subplot(111, projection="3d")
    clean_3d_axes(ax)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_box_aspect((1, 1, 1))
    x_ticks = evenly_spaced_ticks(xlim)
    y_ticks = evenly_spaced_ticks(ylim)
    z_ticks = evenly_spaced_ticks(zlim)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    ax.set_xticklabels([f"{x:.0f}" for x in x_ticks], fontsize=7)
    ax.set_yticklabels([f"{y:.0f}" for y in y_ticks], fontsize=7)
    ax.set_zticklabels([f"{z:.0f}" for z in z_ticks], fontsize=7)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    ax.view_init(elev=8, azim=-90)

    if layout_xs:
        ax.scatter(
            layout_xs,
            layout_ys,
            layout_zs,
            s=4,
            color="lightgray",
            alpha=0.18,
            marker="o",
            depthshade=False,
            linewidths=0,
        )

    sc = None
    if xs:
        sc = ax.scatter(
            xs,
            ys,
            zs,
            c=times,
            s=sizes,
            cmap="viridis",
            marker="o",
            alpha=0.9,
            depthshade=True,
            edgecolors="black",
            linewidths=0.25,
        )
    else:
        ax.text2D(
            0.5,
            0.5,
            "no accepted hits",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="crimson",
        )

    for seg in segments:
        x0, y0, z0 = seg["xi"]
        x1, y1, z1 = seg["xf"]
        color = seg["color"]
        ax.plot([x0, x1], [y0, y1], [z0, z1], color=color, linewidth=2.5, alpha=0.95, label=seg["label"])
        ax.scatter([x0], [y0], [z0], marker="^", s=70, color=color, edgecolor="black", depthshade=False)
        ax.scatter([x1], [y1], [z1], marker="x", s=80, color=color, linewidth=2.0, depthshade=False)

    if segments:
        ax.legend(loc="upper left", fontsize=7)

    if sc is not None:
        fig.colorbar(sc, ax=ax, label="earliest MCPE time [ns]", shrink=0.75)

    def update(angle):
        ax.view_init(elev=8, azim=angle)
        return (sc,) if sc is not None else ()

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(-90, 270, 3),
        interval=70,
        blit=False,
    )

    saved_path = None
    if save:
        saved_path = default_output_path(frame, output_path)
        writer = PillowWriter(fps=fps)
        ani.save(saved_path, writer=writer, dpi=dpi)
        plt.close(fig)
        print("saved:", saved_path)
    elif display_html:
        try:
            from IPython.display import HTML, display

            display(HTML(ani.to_jshtml()))
            plt.close(fig)
        except ImportError:
            print("IPython is not available; returning the animation object without inline display.")

    return fig, ani, saved_path
