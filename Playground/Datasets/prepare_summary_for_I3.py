import os
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


# -------------------------
# Config: sources to scan
# -------------------------
SOURCES = [
    # 98String - I3Photons (*.i3.gz)
    {
        "section": "98String/Muon_I3Photons",
        "dir": "/home/kbas/scratch/98_string/Muon_I3Photons",
        "pattern": "*.i3.gz",
    },
    {
        "section": "98String/Electron_I3Photons",
        "dir": "/home/kbas/scratch/98_string/Electron_I3Photons",
        "pattern": "*.i3.gz",
    },
    {
        "section": "98String/Tau_I3Photons",
        "dir": "/home/kbas/scratch/98_string/Tau_I3Photons",
        "pattern": "*.i3.gz",
    },

    # 340 string - Photon directory (only *.i3)
    {
        "section": "340string/Muon_I3Photons",
        "dir": "/project/6008051/pone_simulation/MC10-000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim/Photon",
        "pattern": "*.i3",
    },

    # Generator directories (*.i3.zst)
    {
        "section": "340string/Electron_I3Photons",
        "dir": "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator",
        "pattern": "*.i3.zst",
    },
    {
        "section": "340string/Tau_I3Photons",
        "dir": "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator",
        "pattern": "*.i3.zst",
    },
]

PHOTON_KEY_DEFAULT = "I3Photons"

# pre-compile regex for speed (micro-optimization)
_SID_RE = re.compile(r"\((\d+)\s*,")


def banner(title: str, char: str = "=") -> str:
    line = char * 90
    return f"\n{line}\n{title}\n{line}\n"


def section_pretty(section: str) -> str:
    parts = section.split("/", 1)
    if len(parts) == 2:
        head, tail = parts
        head_pretty = head.replace("340string", "340 String").replace("98String", "98 String")
        return f"{head_pretty} - {tail}"
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
      - avg_modulekeys_per_nonempty_frame
    """
    unique_strings = set()

    frames_with_key = 0
    frames_with_nonempty = 0
    total_photons = 0

    sum_photons_in_nonempty_frames = 0
    sum_modulekeys_in_nonempty_frames = 0

    f = dataio.I3File(path)
    n_seen = 0
    try:
        while f.more():
            fr = f.pop_frame()
            n_seen += 1
            if max_frames is not None and n_seen > max_frames:
                break

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
                try:
                    sum_modulekeys_in_nonempty_frames += len(phot_map)
                except Exception:
                    pass
    finally:
        f.close()

    avg_ph = (sum_photons_in_nonempty_frames / frames_with_nonempty) if frames_with_nonempty > 0 else 0.0
    avg_keys = (sum_modulekeys_in_nonempty_frames / frames_with_nonempty) if frames_with_nonempty > 0 else 0.0

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
        "avg_modulekeys_per_nonempty_frame": avg_keys,
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
        "avg_modulekeys_per_nonempty_frame": 0.0,
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


def iter_section_jobs(max_files_per_section: Optional[int] = None) -> List[Tuple[str, str, List[str]]]:
    """
    Returns list of (section, base_dir, sorted_files)
    Files are ALWAYS sorted alphabetically (requested).
    """
    out = []
    for src in SOURCES:
        section = src["section"]
        base_dir = src["dir"]
        pattern = os.path.join(base_dir, src["pattern"])
        files = sorted(glob.glob(pattern))  # alphabetic
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
        "avg_modulekeys_per_nonempty_frame",
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

    sections = iter_section_jobs(max_files_per_section=args.max_files_per_section)
    total_files = sum(len(files) for _, _, files in sections)

    print(banner("Scanning I3Photons over multiple directories"))
    print(f"Photon key: {args.photon_key}")
    print(f"Workers: {args.workers}")
    print(f"Total files found: {total_files}")
    print(f"CSV: {args.out_csv}")
    print(f"TXT: {args.out_txt}")
    print("=" * 90, flush=True)

    rows: List[Dict] = []
    n_ok = 0
    n_fail = 0
    done_files = 0

    for section, base_dir, files in sections:
        pretty = section_pretty(section)
        print(banner(f"{pretty} STARTED", char="#"), flush=True)
        print(f"Section path: {base_dir}", flush=True)
        print(f"Files found: {len(files)}", flush=True)

        if not files:
            print(f"{pretty} ... (no files) DONE", flush=True)
            continue

        # Serial mode
        if args.workers <= 1:
            for fp in files:
                row = analyze_one_file(section, fp, args.photon_key, args.max_frames_per_file)

                if row["status"] == "ok":
                    n_ok += 1
                else:
                    n_fail += 1

                rows.append(row)
                done_files += 1

                if (done_files % args.progress_every == 0) or (done_files == total_files):
                    print(f"[{done_files}/{total_files}] processed | ok={n_ok} fail={n_fail}", flush=True)

        # Parallel mode
        else:
            ctx = mp.get_context("spawn")  # safer with IceTray in multiprocess
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
                futures = {
                    ex.submit(analyze_one_file, section, fp, args.photon_key, args.max_frames_per_file): fp
                    for fp in files
                }

                for fut in as_completed(futures):
                    row = fut.result()

                    if row["status"] == "ok":
                        n_ok += 1
                    else:
                        n_fail += 1

                    rows.append(row)
                    done_files += 1

                    if (done_files % args.progress_every == 0) or (done_files == total_files):
                        print(f"[{done_files}/{total_files}] processed | ok={n_ok} fail={n_fail}", flush=True)

        print(f"{pretty} ... ({base_dir}) DONE", flush=True)

    # Deterministic output order (especially for parallel runs)
    rows.sort(key=lambda r: (r.get("section", ""), r.get("file_path", "")))

    write_csv(rows, args.out_csv)
    write_txt_summary(rows, args.out_txt, args.out_csv, started)

    print("\nDONE.", flush=True)
    print(f"CSV: {args.out_csv}", flush=True)
    print(f"TXT: {args.out_txt}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--photon-key", default=PHOTON_KEY_DEFAULT, help="Frame key to analyze (default: I3Photons)")
    ap.add_argument("--out-csv", default="/project/def-nahee/kbas/Graphnet-Applications/Playground/Datasets/summary.csv")
    ap.add_argument("--out-txt", default="/project/def-nahee/kbas/Graphnet-Applications/Playground/Datasets/summary.txt")
    ap.add_argument("--progress-every", type=int, default=50, help="Print progress every N files")
    ap.add_argument("--max-files-per-section", type=int, default=None, help="Debug: limit number of files per section")
    ap.add_argument("--max-frames-per-file", type=int, default=None, help="Debug: limit frames per file (speed)")
    ap.add_argument("--workers", type=int, default=1, help="Parallel file workers (processes)")
    args = ap.parse_args()

    main(args)