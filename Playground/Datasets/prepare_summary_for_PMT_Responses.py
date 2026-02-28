import os
import glob
import re
import time
import csv
import argparse
from typing import Dict, List, Tuple, Optional, Set

from icecube import dataio, icetray  # noqa: F401
from icecube import LeptonInjector  # noqa: F401
from icecube import simclasses      # noqa: F401

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


# -------------------------
# Config: sources to scan (ALL .i3.gz)
# -------------------------
SOURCES = [
    {"section": "102String/Muon_PMT_Response",      "dir": "/scratch/kbas/102_string/Muon_PMT_Response",      "pattern": "*.i3.gz"},
    {"section": "102String/Electron_PMT_Response",  "dir": "/scratch/kbas/102_string/Electron_PMT_Response",  "pattern": "*.i3.gz"},
    {"section": "102String/Tau_PMT_Response",       "dir": "/scratch/kbas/102_string/Tau_PMT_Response",       "pattern": "*.i3.gz"},

    {"section": "340String/Muon_PMT_Response",     "dir": "/scratch/kbas/340_string/Muon_PMT_Response",     "pattern": "*.i3.gz"},
    {"section": "340String/Electron_PMT_Response", "dir": "/scratch/kbas/340_string/Electron_PMT_Response", "pattern": "*.i3.gz"},
    {"section": "340String/Tau_PMT_Response",      "dir": "/scratch/kbas/340_string/Tau_PMT_Response",      "pattern": "*.i3.gz"},
]

EPS_KEY_DEFAULT = "EventPulseSeries"
NONOISE_KEY_DEFAULT = "EventPulseSeries_nonoise"

_SID_RE = re.compile(r"\((\d+)\s*,")


def banner(title: str, char: str = "=") -> str:
    line = char * 90
    return f"\n{line}\n{title}\n{line}\n"


def detect_file_format(path: str) -> str:
    p = path.lower()
    if p.endswith(".i3.gz"):
        return "i3.gz"
    if p.endswith(".i3.zst"):
        return "i3.zst"
    if p.endswith(".i3"):
        return "i3"
    return os.path.splitext(path)[1].lstrip(".")


def get_string_id_from_key(k) -> Optional[int]:
    """OMKey(STRING, OM, PMT) gibi key’lerden string id çıkar."""
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


def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


def analyze_pulse_map_in_file(path: str, key: str, max_frames: Optional[int] = None) -> Dict:
    """
    Bir dosyada ALL frames gezerek, verilen pulse-map key için:
      - frames_with_key
      - frames_with_nonempty (frame total pulses > 0)
      - unique_string_id_count (sadece pulse olan OMKey’ler üzerinden)
      - total_pulse_count (tüm frameler + tüm OMKey’ler)
      - avg_pulses_per_nonempty_frame
      - unique_strings_set (section union için)
    """
    unique_strings: Set[int] = set()
    frames_with_key = 0
    frames_with_nonempty = 0

    total_pulses = 0
    sum_pulses_in_nonempty_frames = 0

    f = dataio.I3File(path)
    seen = 0
    try:
        while f.more():
            fr = f.pop_frame()
            seen += 1
            if max_frames is not None and seen > max_frames:
                break

            if key not in fr:
                continue

            frames_with_key += 1
            pmap = fr[key]

            frame_pulses = 0
            try:
                for omk in pmap.keys():
                    pulses = pmap[omk]
                    n = _safe_len(pulses)
                    frame_pulses += n

                    if n > 0:
                        sid = get_string_id_from_key(omk)
                        if sid is not None:
                            unique_strings.add(sid)
            except Exception:
                # map beklenmedik bir tipteyse bu frame’i pas geç
                pass

            total_pulses += frame_pulses

            if frame_pulses > 0:
                frames_with_nonempty += 1
                sum_pulses_in_nonempty_frames += frame_pulses
    finally:
        f.close()

    avg_pulses = (sum_pulses_in_nonempty_frames / frames_with_nonempty) if frames_with_nonempty > 0 else 0.0

    return {
        "frames_with_key": frames_with_key,
        "frames_with_nonempty": frames_with_nonempty,
        "unique_string_id_count": len(unique_strings),
        "total_pulse_count": total_pulses,
        "avg_pulses_per_nonempty_frame": avg_pulses,
        "unique_strings_set": unique_strings,
    }


def expand_sources(max_files_per_section: Optional[int] = None) -> List[Tuple[str, str]]:
    jobs: List[Tuple[str, str]] = []
    for src in SOURCES:
        section = src["section"]
        pattern = os.path.join(src["dir"], src["pattern"])
        files = sorted(glob.glob(pattern))
        if max_files_per_section is not None:
            files = files[:max_files_per_section]
        for fp in files:
            jobs.append((section, fp))
    return jobs


def analyze_one_job(
    section: str,
    fp: str,
    eps_key: str,
    nonoise_key: str,
    max_frames: Optional[int],
) -> Dict:
    """
    One-file wrapper for multiprocessing (must be top-level).
    Returns a row dict (CSV-ready) + extra sets for TXT union.
    """
    row = {
        "section": section,
        "file_path": fp,
        "file_format": detect_file_format(fp),

        "frames_with_EPS": 0,
        "frames_with_nonempty_EPS": 0,
        "unique_string_id_count_EPS": 0,
        "total_pulse_count_EPS": 0,
        "avg_pulses_per_nonempty_frame_EPS": 0.0,

        "frames_with_EPS_NONOISE": 0,
        "frames_with_nonempty_EPS_NONOISE": 0,
        "unique_string_id_count_EPS_NONOISE": 0,
        "total_pulse_count_EPS_NONOISE": 0,
        "avg_pulses_per_nonempty_frame_EPS_NONOISE": 0.0,

        "status": "ok",
        "error": "",

        # extras (not written to CSV)
        "_eps_strings_set": set(),
        "_nonoise_strings_set": set(),
    }

    try:
        eps = analyze_pulse_map_in_file(fp, key=eps_key, max_frames=max_frames)
        nn  = analyze_pulse_map_in_file(fp, key=nonoise_key, max_frames=max_frames)

        row["frames_with_EPS"] = eps["frames_with_key"]
        row["frames_with_nonempty_EPS"] = eps["frames_with_nonempty"]
        row["unique_string_id_count_EPS"] = eps["unique_string_id_count"]
        row["total_pulse_count_EPS"] = eps["total_pulse_count"]
        row["avg_pulses_per_nonempty_frame_EPS"] = eps["avg_pulses_per_nonempty_frame"]

        row["frames_with_EPS_NONOISE"] = nn["frames_with_key"]
        row["frames_with_nonempty_EPS_NONOISE"] = nn["frames_with_nonempty"]
        row["unique_string_id_count_EPS_NONOISE"] = nn["unique_string_id_count"]
        row["total_pulse_count_EPS_NONOISE"] = nn["total_pulse_count"]
        row["avg_pulses_per_nonempty_frame_EPS_NONOISE"] = nn["avg_pulses_per_nonempty_frame"]

        row["_eps_strings_set"] = eps["unique_strings_set"]
        row["_nonoise_strings_set"] = nn["unique_strings_set"]

    except Exception as e:
        row["status"] = "fail"
        row["error"] = str(e)

    return row


def write_csv(rows: List[Dict], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    fieldnames = [
        "section",
        "file_path",
        "file_format",

        "frames_with_EPS",
        "frames_with_nonempty_EPS",
        "frames_with_EPS_NONOISE",
        "frames_with_nonempty_EPS_NONOISE",

        "unique_string_id_count_EPS",
        "unique_string_id_count_EPS_NONOISE",
        "total_pulse_count_EPS",
        "total_pulse_count_EPS_NONOISE",
        "avg_pulses_per_nonempty_frame_EPS",
        "avg_pulses_per_nonempty_frame_EPS_NONOISE",

        "status",
        "error",
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _write_wrapped_list(fh, prefix: str, values: List[int], wrap: int = 120) -> None:
    s = ", ".join(str(x) for x in values)
    if not s:
        fh.write(prefix + "(empty)\n")
        return

    line = prefix
    for token in s.split(", "):
        add = token if line.endswith(prefix) else ", " + token
        if len(line) + len(add) > wrap:
            fh.write(line + "\n")
            line = " " * len(prefix) + token
        else:
            line += add
    fh.write(line + "\n")


def write_txt_summary(
    rows: List[Dict],
    out_txt: str,
    out_csv: str,
    started: float,
    section_union_eps: Dict[str, Set[int]],
    section_union_nonoise: Dict[str, Set[int]],
) -> None:
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    fail_rows = [r for r in rows if r.get("status") != "ok"]
    elapsed = time.time() - started

    # per-section aggregates (OK only)
    per_section = {}
    for r in ok_rows:
        sec = r["section"]
        per_section.setdefault(sec, {
            "n": 0,
            "sum_eps": 0,
            "sum_nonoise": 0,
        })
        a = per_section[sec]
        a["n"] += 1
        a["sum_eps"] += int(r.get("total_pulse_count_EPS", 0) or 0)
        a["sum_nonoise"] += int(r.get("total_pulse_count_EPS_NONOISE", 0) or 0)

    with open(out_txt, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("PMT Response summary (EventPulseSeries vs EventPulseSeries_nonoise)\n")
        f.write("=" * 90 + "\n")
        f.write(f"CSV output: {out_csv}\n")
        f.write(f"Scanned files (total): {len(rows)}\n")
        f.write(f"OK: {len(ok_rows)}\n")
        f.write(f"FAIL: {len(fail_rows)}\n")
        f.write(f"Elapsed seconds: {elapsed:.1f}\n\n")

        f.write("Per-section aggregates (OK files only):\n")
        f.write(
            f"{'section':38s}  {'n_files':>7s}  {'avg_eps_pulses':>14s}  {'avg_nonoise_pulses':>18s}\n"
        )
        f.write("-" * 90 + "\n")
        for sec, a in per_section.items():
            n = a["n"]
            avg_eps = (a["sum_eps"] / n) if n else 0.0
            avg_nonoise = (a["sum_nonoise"] / n) if n else 0.0
            f.write(
                f"{sec:38s}  {n:7d}  {avg_eps:14.2f}  {avg_nonoise:18.2f}\n"
            )

        f.write("\n" + "=" * 90 + "\n")
        f.write("Section-wise UNION of unique string IDs (over ALL files)\n")
        f.write("=" * 90 + "\n")

        all_secs = sorted(set(list(section_union_eps.keys()) + list(section_union_nonoise.keys())))
        for sec in all_secs:
            eps_list = sorted(section_union_eps.get(sec, set()))
            nn_list = sorted(section_union_nonoise.get(sec, set()))

            f.write(f"\n[{sec}]\n")
            f.write(f"  EPS union unique strings count: {len(eps_list)}\n")
            _write_wrapped_list(f, "  EPS strings: ", eps_list)

            f.write(f"  NONOISE union unique strings count: {len(nn_list)}\n")
            _write_wrapped_list(f, "  NONOISE strings: ", nn_list)

        if fail_rows:
            f.write("\nFailures (first 20):\n")
            for r in fail_rows[:20]:
                f.write(f"- {r.get('file_path')} | {r.get('error')}\n")


def main(args):
    started = time.time()
    jobs = expand_sources(max_files_per_section=args.max_files_per_section)
    total_files = len(jobs)

    print(banner("Scanning PMT Response (EPS vs NONOISE) over multiple directories"))
    print(f"EPS key: {args.eps_key}")
    print(f"NONOISE key: {args.nonoise_key}")
    print(f"Workers: {args.workers}")
    print(f"Total files found: {total_files}")
    print(f"CSV: {args.out_csv}")
    print(f"TXT: {args.out_txt}")
    print("=" * 90, flush=True)

    rows: List[Dict] = []
    n_ok = 0
    n_fail = 0
    done = 0

    # For TXT: union of unique strings per section
    section_union_eps: Dict[str, Set[int]] = {}
    section_union_nonoise: Dict[str, Set[int]] = {}

    if args.workers <= 1:
        for section, fp in jobs:
            row = analyze_one_job(section, fp, args.eps_key, args.nonoise_key, args.max_frames_per_file)

            if row["status"] == "ok":
                n_ok += 1
                section_union_eps.setdefault(section, set()).update(row.get("_eps_strings_set", set()))
                section_union_nonoise.setdefault(section, set()).update(row.get("_nonoise_strings_set", set()))
            else:
                n_fail += 1

            rows.append(row)
            done += 1
            if (done % args.progress_every == 0) or (done == total_files):
                print(f"[{done}/{total_files}] processed | ok={n_ok} fail={n_fail}", flush=True)

    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futures = {
                ex.submit(analyze_one_job, section, fp, args.eps_key, args.nonoise_key, args.max_frames_per_file): (section, fp)
                for (section, fp) in jobs
            }

            for fut in as_completed(futures):
                section, fp = futures[fut]
                row = fut.result()

                if row["status"] == "ok":
                    n_ok += 1
                    section_union_eps.setdefault(section, set()).update(row.get("_eps_strings_set", set()))
                    section_union_nonoise.setdefault(section, set()).update(row.get("_nonoise_strings_set", set()))
                else:
                    n_fail += 1

                rows.append(row)
                done += 1
                if (done % args.progress_every == 0) or (done == total_files):
                    print(f"[{done}/{total_files}] processed | ok={n_ok} fail={n_fail}", flush=True)

    # deterministic output order
    rows.sort(key=lambda r: (r.get("section", ""), r.get("file_path", "")))

    write_csv(rows, args.out_csv)
    write_txt_summary(rows, args.out_txt, args.out_csv, started, section_union_eps, section_union_nonoise)

    print("\nDONE.", flush=True)
    print(f"CSV: {args.out_csv}", flush=True)
    print(f"TXT: {args.out_txt}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps-key", default=EPS_KEY_DEFAULT, help="noise+signal pulse key (default: EventPulseSeries)")
    ap.add_argument("--nonoise-key", default=NONOISE_KEY_DEFAULT, help="signal-only pulse key (default: EventPulseSeries_nonoise)")
    ap.add_argument("--out-csv", default="/project/def-nahee/kbas/Graphnet-Applications/Playground/Datasets/summary_pmt.csv")
    ap.add_argument("--out-txt", default="/project/def-nahee/kbas/Graphnet-Applications/Playground/Datasets/summary_pmt.txt")
    ap.add_argument("--max-files-per-section", type=int, default=None, help="Debug: limit number of files per section")
    ap.add_argument("--max-frames-per-file", type=int, default=None, help="Debug: limit frames per file (speed)")
    ap.add_argument("--workers", type=int, default=1, help="Parallel file workers (processes)")
    ap.add_argument("--progress-every", type=int, default=50, help="Print progress every N files")
    args = ap.parse_args()

    main(args)