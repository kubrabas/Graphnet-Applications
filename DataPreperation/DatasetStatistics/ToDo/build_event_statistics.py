#!/usr/bin/env python3
"""
Build an event-level CSV for reproducible DatasetStatistics plots.

Usage:

    /home/kbas/SlurmScripts/DataPreperation/submit_event_statistics.sh \
      --mc-name STRING340MC \
      --geometry full_geometry \
      --workers 8
"""

import argparse
import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from icecube import icetray, dataio, dataclasses, LeptonInjector, simclasses  # noqa: F401

icetray.I3Logger.global_logger = icetray.I3NullLogger()

sys.path.insert(0, "/project/def-nahee/kbas/Graphnet-Applications/Metadata")
import paths  # noqa: E402


MC_DATASETS = {
    "SPRING2026MC": paths.SPRING2026MC_I3,
    "STRING340MC": paths.STRING340MC_I3,
}

MC_FOLDERS = {
    "SPRING2026MC": "Spring2026MC",
    "STRING340MC": "String340MC",
}

OUTPUT_BASE = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/DatasetStatistics"
PULSEMAP_KEY_DEFAULT = "Accepted_PulseMap"

FIELDNAMES = [
    "path",
    "flavor",
    "geometry",
    "daq_index",
    "RunID",
    "SubrunID",
    "EventID",
    "SubEventID",
    "accepted_pulse_count",
    "position_x",
    "position_y",
    "position_z",
    "pid",
    "interaction_type",
    "totalEnergy",
    "zenith",
    "azimuth",
    "finalStateX",
    "finalStateY",
    "finalType1",
    "finalType2",
    "initialType",
    "totalColumnDepth",
    "impactParameter",
    "three_category_label",
    "is_track",
    "status",
    "error",
]


def get_datasets(mc_name, geometry):
    return {
        flavor: info
        for flavor, info in MC_DATASETS[mc_name][geometry].items()
        if info.get("path") is not None and info.get("format") is not None
    }


def bad_file_rule(fpath):
    target = os.path.abspath(fpath)
    bad_i3_files = getattr(paths, "BAD_I3_FILES", {})

    for flavors in bad_i3_files.values():
        for info in flavors.values():
            no_daq = set(map(os.path.abspath, map(str, info.get("no_daq_for_some_reason", set()))))
            if target in no_daq:
                return "skip_no_daq", None

            for bad_path, daq_count in info.get("available_daq_counts", {}).items():
                if target == os.path.abspath(str(bad_path)):
                    return "limit_daq", int(daq_count)

    return "normal", None


def pattern_for_format(fmt):
    return "*.i3" if fmt == "i3" else f"*.i3.{fmt}"


def count_pulses(frame, pulsemap_key):
    if pulsemap_key not in frame:
        return ""
    pulsemap = frame[pulsemap_key]
    return sum(len(pulses) for pulses in pulsemap.values())


def interaction_type(ep):
    neutrinos = [
        dataclasses.I3Particle.NuE,
        dataclasses.I3Particle.NuMu,
        dataclasses.I3Particle.NuTau,
        dataclasses.I3Particle.NuEBar,
        dataclasses.I3Particle.NuMuBar,
        dataclasses.I3Particle.NuTauBar,
    ]
    return 2 if ep.finalType1 in neutrinos else 1


def three_category_label(pid, int_type):
    if int_type == 2 or abs(pid) == 12:
        return "cascade"
    if abs(pid) == 14 and int_type == 1:
        return "muon_CC"
    if abs(pid) == 16 and int_type == 1:
        return "tau_CC"
    return "unknown"


def event_row(fpath, flavor, geometry, daq_index, frame, pulsemap_key):
    row = {name: "" for name in FIELDNAMES}
    row.update({
        "path": fpath,
        "flavor": flavor,
        "geometry": geometry,
        "daq_index": daq_index,
        "status": "ok",
    })

    if "I3EventHeader" in frame:
        header = frame["I3EventHeader"]
        row.update({
            "RunID": header.run_id,
            "SubrunID": header.sub_run_id,
            "EventID": header.event_id,
            "SubEventID": header.sub_event_id,
        })

    row["accepted_pulse_count"] = count_pulses(frame, pulsemap_key)

    if "EventProperties" not in frame:
        row["status"] = "missing_event_properties"
        row["error"] = "EventProperties missing"
        return row

    ep = frame["EventProperties"]
    int_type = interaction_type(ep)
    pid = int(ep.initialType)

    row.update({
        "position_x": ep.x,
        "position_y": ep.y,
        "position_z": ep.z,
        "pid": pid,
        "interaction_type": int_type,
        "totalEnergy": ep.totalEnergy,
        "zenith": ep.zenith,
        "azimuth": ep.azimuth,
        "finalStateX": ep.finalStateX,
        "finalStateY": ep.finalStateY,
        "finalType1": int(ep.finalType1),
        "finalType2": int(ep.finalType2),
        "initialType": pid,
        "three_category_label": three_category_label(pid, int_type),
        "is_track": int(abs(pid) == 14 and int_type == 1),
    })

    for attr in ("totalColumnDepth", "impactParameter"):
        try:
            row[attr] = getattr(ep, attr)
        except AttributeError:
            pass

    return row


def rows_for_file(fpath, flavor, geometry, pulsemap_key):
    rows = []
    messages = []
    rule, max_daq = bad_file_rule(fpath)
    if rule == "skip_no_daq":
        messages.append(f"  [skip bad file] {fpath}")
        return rows, messages

    f = None
    daq_index = -1
    try:
        f = dataio.I3File(fpath)
        while f.more():
            frame = f.pop_frame()
            if frame.Stop != icetray.I3Frame.DAQ:
                continue

            daq_index += 1
            if rule == "limit_daq" and daq_index >= max_daq:
                break

            rows.append(event_row(fpath, flavor, geometry, daq_index, frame, pulsemap_key))
            if rule == "limit_daq" and daq_index + 1 >= max_daq:
                break
    except Exception as e:
        messages.append(f"  [bad file] {fpath} -- {e}")
    finally:
        if f is not None:
            f.close()

    return rows, messages


def process_file(job):
    flavor, geometry, pulsemap_key, fpath = job
    rows, messages = rows_for_file(fpath, flavor, geometry, pulsemap_key)
    return flavor, rows, messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-name", default="SPRING2026MC", choices=list(MC_DATASETS))
    ap.add_argument("--geometry", default="full_geometry")
    ap.add_argument("--pulsemap-key", default=PULSEMAP_KEY_DEFAULT)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    datasets = get_datasets(args.mc_name, args.geometry)
    out_dir = os.path.join(OUTPUT_BASE, MC_FOLDERS[args.mc_name], args.geometry)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = args.out_csv or os.path.join(out_dir, "event_statistics.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        total_rows = 0
        for flavor, info in datasets.items():
            pattern = os.path.join(info["path"], "**", pattern_for_format(info["format"]))
            files = sorted(glob.glob(pattern, recursive=True))
            print(f"\n=== {flavor}: {len(files)} files ===")

            flavor_rows = 0
            jobs = [(flavor, args.geometry, args.pulsemap_key, fpath) for fpath in files]
            if args.workers <= 1:
                results = map(process_file, jobs)
            else:
                executor = ProcessPoolExecutor(max_workers=args.workers)
                results = executor.map(process_file, jobs, chunksize=1)

            try:
                for _, rows, messages in results:
                    for msg in messages:
                        print(msg)
                    for row in rows:
                        writer.writerow(row)
                        flavor_rows += 1
                        total_rows += 1
            finally:
                if args.workers > 1:
                    executor.shutdown()

            print(f"  rows written: {flavor_rows}")

    print(f"\nSaved: {out_csv}")
    print(f"Total rows: {total_rows}")


if __name__ == "__main__":
    main()
