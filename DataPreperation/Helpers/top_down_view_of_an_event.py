"""Top-down event display helper for accepted pulse maps.

Example
-------
Load the first frame with non-empty accepted pulses and draw a larger top-down
view. The detector string layout is read from the string-coordinate CSV.

    from pathlib import Path
    from icecube import dataio, LeptonInjector, simclasses
    from top_down_view_of_an_event import top_down_view_of_an_event

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

    top_down_view_of_an_event(
        frame,
        pulsemap_key=pulsemap_key,
        title="tau_gen_1212 first accepted-pulse frame: top-down view",
        figsize=(10, 8.5),
        max_string_labels=12,
    )
"""

from collections import defaultdict
from pathlib import Path
import csv

import matplotlib.pyplot as plt

DEFAULT_PULSEMAP_KEY = "Accepted_PulseMap_340String"
DEFAULT_STRING_COORDS_CSV = Path(
    "/project/def-nahee/kbas/Graphnet-Applications/Metadata/GeometryFiles/string_coordinates_340_string_mc.csv"
)


def read_string_coords(csv_path=DEFAULT_STRING_COORDS_CSV):
    string_coords = {}
    with Path(csv_path).open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            string_coords[int(row["string"])] = (float(row["x"]), float(row["y"]))
    return string_coords


def pe_charge(pe):
    if hasattr(pe, "npe"):
        return float(pe.npe)
    if hasattr(pe, "NPE"):
        return float(pe.NPE)
    return 1.0


def pe_time(pe):
    if hasattr(pe, "time"):
        return float(pe.time)
    return float(pe.Time)


def pulsemap_to_string_hits(frame, string_coords, pulsemap_key=DEFAULT_PULSEMAP_KEY):
    if pulsemap_key not in frame:
        print(f"{pulsemap_key} not found in frame; no accepted hits to plot.")
        return [], [], [], [], []

    pulse_map = frame[pulsemap_key]
    if not hasattr(pulse_map, "items"):
        raise TypeError(
            f"{pulsemap_key} is not a typed map. "
            "Restart the kernel and run this after importing simclasses."
        )

    if not pulse_map:
        print(f"{pulsemap_key} is empty; no accepted hits to plot.")
        return [], [], [], [], []

    charge_by_string = defaultdict(float)
    time_by_string = {}

    for omkey, pulses in pulse_map.items():
        string = int(omkey.string)
        for pe in pulses:
            charge_by_string[string] += pe_charge(pe)
            t = pe_time(pe)
            if string not in time_by_string or t < time_by_string[string]:
                time_by_string[string] = t

    detected_strings = sorted(
        string
        for string in charge_by_string
        if charge_by_string[string] > 0 and string in string_coords
    )

    det_x = [string_coords[string][0] for string in detected_strings]
    det_y = [string_coords[string][1] for string in detected_strings]
    det_t = [time_by_string[string] for string in detected_strings]
    det_s = [25 + 6 * min(charge_by_string[string], 20) for string in detected_strings]

    if not detected_strings:
        print(f"{pulsemap_key} has no accepted hits with known string coordinates.")

    return detected_strings, det_x, det_y, det_t, det_s


def default_output_path(frame, output_path):
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "top_down_view_of_an_event.png"
    else:
        output_path = Path(output_path)

    if output_path.suffix.lower() not in {".png", ".pdf", ".jpg", ".jpeg", ".svg"}:
        run_id = None
        event_id = None
        if "I3EventHeader" in frame:
            header = frame["I3EventHeader"]
            run_id = int(getattr(header, "run_id", 0))
            event_id = int(getattr(header, "event_id", 0))
        name = f"run{run_id:04d}_event{event_id:03d}_top_down.png" if run_id or event_id else "top_down_view_of_an_event.png"
        output_path = output_path / name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def top_down_view_of_an_event(
    frame,
    string_coords=None,
    string_coords_csv=DEFAULT_STRING_COORDS_CSV,
    pulsemap_key=DEFAULT_PULSEMAP_KEY,
    title=None,
    vmin=None,
    vmax=None,
    add_colorbar=True,
    set_axis_labels=True,
    save=False,
    output_path=None,
    ax=None,
    figsize=(8.5, 7.5),
    annotate_strings=True,
    max_string_labels=80,
):
    if string_coords is None:
        string_coords = read_string_coords(string_coords_csv)

    all_string_x = [xy[0] for xy in string_coords.values()]
    all_string_y = [xy[1] for xy in string_coords.values()]
    detected_strings, det_x, det_y, det_t, det_s = pulsemap_to_string_hits(
        frame,
        string_coords,
        pulsemap_key=pulsemap_key,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0.10, right=0.86, bottom=0.10, top=0.90)
    else:
        fig = ax.figure

    ax.scatter(all_string_x, all_string_y, s=12, color="lightgray", alpha=0.35)

    mappable = None
    if det_x:
        if vmin is None:
            vmin = min(det_t)
        if vmax is None:
            vmax = max(det_t)
        mappable = ax.scatter(
            det_x,
            det_y,
            s=det_s,
            c=det_t,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            edgecolor="black",
            linewidth=0.55,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "no accepted hits",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="crimson",
        )

    if annotate_strings and len(detected_strings) <= max_string_labels:
        for string, x, y in zip(detected_strings, det_x, det_y):
            ax.text(x, y, str(string), fontsize=6, ha="center", va="bottom")
    elif annotate_strings:
        print(f"Skipping string labels: {len(detected_strings)} detected strings exceeds max_string_labels={max_string_labels}.")

    if title is None:
        if "I3EventHeader" in frame:
            header = frame["I3EventHeader"]
            title = f"Run {int(header.run_id)} / Event {int(header.event_id)}"
        else:
            title = "Top-down event view"

    ax.set_title(title, fontsize=11, pad=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    if set_axis_labels:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    if add_colorbar and mappable is not None:
        fig.colorbar(mappable, ax=ax, label="earliest MCPE time [ns]", pad=0.035, fraction=0.046)

    saved_path = None
    if save:
        saved_path = default_output_path(frame, output_path)
        fig.savefig(saved_path, dpi=150, bbox_inches="tight")
        print("saved:", saved_path)

    return fig, ax, saved_path
