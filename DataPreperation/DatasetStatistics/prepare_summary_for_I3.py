import os
import sys
import glob
import re
import time
import csv
import argparse
from typing import Dict, List, Tuple, Optional

from icecube import dataio, icetray  # noqa: F401
from icecube import LeptonInjector  # noqa: F401
from icecube import simclasses      # noqa: F401

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/project/def-nahee/kbas/Graphnet-Applications/Metadata")
import paths as _paths  # noqa: E402

PHOTON_KEY_DEFAULT = "I3Photons"

_FORMAT_PATTERN = {
    "zst": "*.i3.zst",
    "gz":  "*.i3.gz",
    "i3":  "*.i3",
}

MC_DATASETS = {
    "SPRING2026MC": _paths.SPRING2026MC_I3,
    "STRING340MC":  _paths.STRING340MC_I3,
}


def build_sources(mc_name: str) -> List[Dict]:
    """
    Build the source list from Metadata/paths.py for the given MC_NAME.
    Only entries where both path and format are not None are included.
    """
    dataset = MC_DATASETS.get(mc_name)
    if dataset is None:
        raise ValueError(f"Unknown MC_NAME '{mc_name}'. Available: {list(MC_DATASETS)}")

    sources = []
    for geometry, flavors in dataset.items():
        for flavor, info in flavors.items():
            if info.get("path") is None or info.get("format") is None:
                continue
            fmt = info["format"]
            pattern = _FORMAT_PATTERN.get(fmt, f"*.i3.{fmt}")
            sources.append({
                "section": f"{geometry}/{flavor}",
                "dir":     info["path"],
                "pattern": pattern,
            })
    return sources

# pre-compile regex for speed (micro-optimization)
_SID_RE = re.compile(r"\((\d+)\s*,")


def banner(title: str, char: str = "=") -> str:
    line = char * 90
    return f"\n{line}\n{title}\n{line}\n"


def section_pretty(section: str) -> str:
    parts = section.split("/", 1)
    if len(parts) == 2:
        geometry, flavor = parts
        return f"{geometry} - {flavor}"
    return section


def detect_file_format(path: str) -> str:
    p = path.lower()
    if p.endswith(".i3.zst"):
        return "i3.zst"
    if p.endswith(".i3.gz"):
        return "i3.gz"
    if p.endswith(".i3"):
        return "i3"
    return os.path.splitext(path)[1].lstrip(".")


def get_string_id_from_key(k) -> Optional[int]:
    """
    Keys often look like ModuleKey(STRING, MODULE) or OMKey(STRING, OM).
    Try attribute access, then tuple, then parse from string.
    """
    if hasattr(k, "string"):
        try:
            return int(k.string)
        except Exception:
            try:
                return int(k.string())
            except Exception:
                pass

    if isinstance(k, tuple) and len(k) > 0:
        return get_string_id_from_key(k[0])

    s = str(k)
    m = _SID_RE.search(s)
    if m:
        return int(m.group(1))

    return None


def analyze_file(path: str, photon_key: str, max_frames: Optional[int] = None) -> Dict:
    """
    Scans ALL frames (or first max_frames if set).

    Metrics:
      - frames_with_I3Photons
      - frames_with_nonempty_I3Photons
      - total_photon_count
      - unique_string_ids: sorted list of unique string IDs that have >=1 photon somewhere
      - unique_string_id_count: len(unique_string_ids)
      - avg_photons_per_nonempty_frame
    """
    unique_strings = set()

    frames_with_key = 0
    frames_with_nonempty = 0
    total_photons = 0

    sum_photons_in_nonempty_frames = 0

    f = dataio.I3File(path)
    n_seen = 0
    try:
        while f.more():
            fr = f.pop_frame()
            n_seen += 1
            if max_frames is not None and n_seen > max_frames:
                break

            if fr.Stop != icetray.I3Frame.DAQ:
                continue

            if photon_key not in fr:
                continue

            frames_with_key += 1
            phot_map = fr[photon_key]

            frame_photons = 0
            try:
                for k in phot_map.keys():
                    try:
                        series = phot_map[k]
                        nph = len(series)
                    except Exception:
                        nph = 0

                    frame_photons += nph

                    if nph > 0:
                        sid = get_string_id_from_key(k)
                        if sid is not None:
                            unique_strings.add(sid)
            except Exception:
                pass

            total_photons += frame_photons

            if frame_photons > 0:
                frames_with_nonempty += 1
                sum_photons_in_nonempty_frames += frame_photons

    finally:
        f.close()

    avg_ph = (sum_photons_in_nonempty_frames / frames_with_nonempty) if frames_with_nonempty > 0 else 0.0

    unique_list = sorted(unique_strings)

    return {
        "file_format": detect_file_format(path),
        "data_format": photon_key,
        "frames_with_I3Photons": frames_with_key,
        "frames_with_nonempty_I3Photons": frames_with_nonempty,
        "unique_string_id_count": len(unique_list),
        "unique_string_ids": unique_list,
        "total_photon_count": total_photons,
        "avg_photons_per_nonempty_frame": avg_ph,
    }


def analyze_one_file(section: str, fp: str, photon_key: str, max_frames: Optional[int]) -> Dict:
    """
    One-file wrapper for multiprocessing. MUST be top-level (picklable).
    Returns the same row dict you were writing before.
    """
    row = {
        "section": section,
        "file_path": fp,
        "file_format": detect_file_format(fp),
        "data_format": photon_key,
        "frames_with_I3Photons": 0,
        "frames_with_nonempty_I3Photons": 0,
        "unique_string_id_count": 0,
        "total_photon_count": 0,
        "avg_photons_per_nonempty_frame": 0.0,
        "status": "ok",
        "error": "",
        # not written to CSV, but used for TXT union lists
        "unique_string_ids": [],
    }

    try:
        stats = analyze_file(fp, photon_key=photon_key, max_frames=max_frames)
        row.update(stats)
    except Exception as e:
        row["status"] = "fail"
        row["error"] = str(e)

    return row


def iter_section_jobs(mc_name: str, max_files_per_section: Optional[int] = None) -> List[Tuple[str, str, List[str]]]:
    """
    Returns list of (section, base_dir, sorted_files).
    Files are sorted alphabetically.
    """
    sources = build_sources(mc_name)
    out = []
    for src in sources:
        section = src["section"]
        base_dir = src["dir"]
        pattern = os.path.join(base_dir, "**", src["pattern"])
        files = sorted(glob.glob(pattern, recursive=True))
        if max_files_per_section is not None:
            files = files[:max_files_per_section]
        out.append((section, base_dir, files))
    return out


def write_csv(rows: List[Dict], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    fieldnames = [
        "section",
        "file_path",
        "file_format",
        "data_format",
        "frames_with_I3Photons",
        "frames_with_nonempty_I3Photons",
        "unique_string_id_count",
        "total_photon_count",
        "avg_photons_per_nonempty_frame",
        "status",
        "error",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _wrap_int_list(ints: List[int], width: int = 90, prefix: str = "  ") -> str:
    """
    Pretty print list of ints as comma-separated lines.
    """
    s = ", ".join(str(x) for x in ints)
    if not s:
        return prefix + "(none)\n"

    lines = []
    cur = ""
    for token in s.split(", "):
        if not cur:
            cur = token
        elif len(cur) + 2 + len(token) <= width - len(prefix):
            cur += ", " + token
        else:
            lines.append(prefix + cur)
            cur = token
    if cur:
        lines.append(prefix + cur)
    return "\n".join(lines) + "\n"


def write_txt_summary(rows: List[Dict], out_txt: str, out_csv: str, started: float) -> None:
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    fail_rows = [r for r in rows if r.get("status") != "ok"]

    elapsed = time.time() - started

    # Per-section aggregates (OK files only) with UNION unique strings
    per_section: Dict[str, Dict] = {}
    for r in ok_rows:
        sec = r["section"]
        per_section.setdefault(sec, {
            "n_files": 0,
            "sum_photons": 0,
            "sum_frames_with_key": 0,
            "sum_nonempty_frames": 0,
            "unique_strings_union": set(),
        })
        a = per_section[sec]
        a["n_files"] += 1
        a["sum_photons"] += int(r.get("total_photon_count", 0) or 0)
        a["sum_frames_with_key"] += int(r.get("frames_with_I3Photons", 0) or 0)
        a["sum_nonempty_frames"] += int(r.get("frames_with_nonempty_I3Photons", 0) or 0)

        ids = r.get("unique_string_ids", []) or []
        try:
            a["unique_strings_union"].update(ids)
        except Exception:
            pass

    with open(out_txt, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("I3Photons summary\n")
        f.write("=" * 90 + "\n")
        f.write(f"CSV output: {out_csv}\n")
        f.write(f"Scanned files (total): {len(rows)}\n")
        f.write(f"OK: {len(ok_rows)}\n")
        f.write(f"FAIL: {len(fail_rows)}\n")
        f.write(f"Elapsed seconds: {elapsed:.1f}\n\n")

        f.write("Per-section totals (OK files only):\n")
        f.write(
            f"{'section':40s}  {'n_files':>8s}  {'sum_photons':>14s}  "
            f"{'sum_frames_with_key':>20s}  {'sum_nonempty_frames':>20s}  "
            f"{'unique_strings_union':>20s}\n"
        )
        f.write("-" * 90 + "\n")

        for sec, a in per_section.items():
            uniq_list = sorted(a["unique_strings_union"])
            f.write(
                f"{sec:40s}  {a['n_files']:8d}  {a['sum_photons']:14d}  "
                f"{a['sum_frames_with_key']:20d}  {a['sum_nonempty_frames']:20d}  "
                f"{len(uniq_list):20d}\n"
            )

        f.write("\n")
        f.write("=" * 90 + "\n")
        f.write("Per-section UNIQUE string ID lists (UNION over all scanned files)\n")
        f.write("=" * 90 + "\n")

        for sec, a in per_section.items():
            uniq_list = sorted(a["unique_strings_union"])
            f.write(f"\n[{sec}] unique_string_id_count = {len(uniq_list)}\n")
            f.write(_wrap_int_list(uniq_list, width=90, prefix="  "))

        if fail_rows:
            f.write("\n")
            f.write("=" * 90 + "\n")
            f.write("Failures (first 20):\n")
            f.write("=" * 90 + "\n")
            for r in fail_rows[:20]:
                f.write(f"- {r.get('file_path')} | {r.get('error')}\n")


def main(args):
    started = time.time()

    sections = iter_section_jobs(args.mc_name, max_files_per_section=args.max_files_per_section)
    total_files = sum(len(files) for _, _, files in sections)

    print(banner("Scanning I3Photons over multiple directories"))
    print(f"MC dataset: {args.mc_name}")
    print(f"Photon key: {args.photon_key}")
    print(f"Workers: {args.workers}")
    print(f"Total files found: {total_files}")
    print(f"CSV: {args.out_csv}")
    print(f"TXT: {args.out_txt}")
    print("=" * 90, flush=True)

    rows: List[Dict] = []

    for section, base_dir, files in sections:
        pretty = section_pretty(section)
        print(banner(f"{pretty} STARTED", char="#"), flush=True)
        print(f"Section path: {base_dir}", flush=True)
        print(f"Files found: {len(files)}", flush=True)

        if not files:
            print(f"{pretty} DONE (no files)", flush=True)
            continue

        sec_done = 0
        sec_ok = 0
        sec_fail = 0

        def _record(row):
            nonlocal sec_done, sec_ok, sec_fail
            rows.append(row)
            sec_done += 1
            if row["status"] == "ok":
                sec_ok += 1
            else:
                sec_fail += 1
            if sec_done % args.progress_every == 0 or sec_done == len(files):
                print(f"  [{sec_done}/{len(files)}] ok={sec_ok} fail={sec_fail}", flush=True)

        if args.workers <= 1:
            for fp in files:
                _record(analyze_one_file(section, fp, args.photon_key, args.max_frames_per_file))
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
                futures = {
                    ex.submit(analyze_one_file, section, fp, args.photon_key, args.max_frames_per_file): fp
                    for fp in files
                }
                for fut in as_completed(futures):
                    _record(fut.result())

        print(f"{pretty} DONE | files={len(files)} ok={sec_ok} fail={sec_fail}", flush=True)

    # Deterministic output order (especially for parallel runs)
    rows.sort(key=lambda r: (r.get("section", ""), r.get("file_path", "")))

    write_csv(rows, args.out_csv)
    write_txt_summary(rows, args.out_txt, args.out_csv, started)

    print("\nDONE.", flush=True)
    print(f"CSV: {args.out_csv}", flush=True)
    print(f"TXT: {args.out_txt}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-name", required=True, choices=list(MC_DATASETS), help=f"MC dataset to scan. Choices: {list(MC_DATASETS)}")
    ap.add_argument("--photon-key", default=PHOTON_KEY_DEFAULT, help="Frame key to analyze (default: I3Photons)")
    ap.add_argument("--out-csv", default="/project/def-nahee/kbas/Graphnet-Applications/Playground/Datasets/summary_i3.csv")
    ap.add_argument("--out-txt", default="/project/def-nahee/kbas/Graphnet-Applications/Playground/Datasets/summary_i3.txt")
    ap.add_argument("--progress-every", type=int, default=50, help="Print progress every N files")
    ap.add_argument("--max-files-per-section", type=int, default=None, help="Debug: limit number of files per section")
    ap.add_argument("--max-frames-per-file", type=int, default=None, help="Debug: limit frames per file (speed)")
    ap.add_argument("--workers", type=int, default=1, help="Parallel file workers (processes)")
    args = ap.parse_args()

    main(args)