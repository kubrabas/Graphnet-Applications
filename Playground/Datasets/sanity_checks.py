#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sanity_checks.py

Print a clean summary of how many files ("batches") exist under key dataset folders,
and how those files are distributed by format (i3, i3.gz, i3.zst, zst, parquet, etc.).

UPDATED:
- Also writes the same terminal output to a timestamped .txt file (Europe/Berlin).
- Section 1: "340 String - I3Photons" (muon/electron/tau simulation folders).
- Section 2: "340 String - PMT Response" (local scratch PMT response folders).
- Section 3: "98 String - I3Photons" (Raw_ folders).
- Removed the top global header block.
- ADDED: Summary table at the very end (rows: Muon/Tau/Electron, columns: 340 I3Photons / 340 PMT Response / 98 I3Photons),
         with I3-like totals in the cells.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from contextlib import redirect_stdout

try:
    from zoneinfo import ZoneInfo  # py3.9+
except ImportError:
    ZoneInfo = None


# =============================================================================
# 0) Output log path
# =============================================================================

LOG_DIR = "/project/def-nahee/kbas/Graphnet-Applications/Playground/Datasets"
BERLIN_TZ = "Europe/Berlin"


# =============================================================================
# 1) Paths
# =============================================================================

############ Section 1: 340 String - I3Photons ############
MUON_I3PHOTONS = "/project/6008051/pone_simulation/MC10-000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim/Photon"
ELECTRON_I3PHOTONS = "/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator"
TAU_I3PHOTONS = "/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator"

############ Section 2: 340 String - PMT Response (local scratch) ############
MAIN_340 = "/home/kbas/scratch/340_string"
MUON_REL_340 = "PMT_RESPONSE_Muon"
ELECTRON_REL_340 = "PMT_RESPONSE_Electron"
TAU_REL_340 = "PMT_RESPONSE_Tau"

############ Section 3: 98 String - I3Photons (Raw_ folders; local scratch) ############
MAIN_98 = "/home/kbas/scratch/98_string"
RAW_MUON_REL_98 = "Raw_Muon"
RAW_ELECTRON_REL_98 = "Raw_Electron"


# =============================================================================
# 2) Small Utilities
# =============================================================================

def berlin_timestamp_str() -> str:
    """Return timestamp string as DD_MM_YYYY_HH_MM_SS in Europe/Berlin timezone."""
    if ZoneInfo is not None:
        tz = ZoneInfo(BERLIN_TZ)
        now = datetime.now(tz)
    else:
        now = datetime.now()
    return now.strftime("%d_%m_%Y_%H_%M_%S")


class Tee:
    """File-like object that duplicates writes to multiple streams."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def classify_format(p: Path) -> str:
    """Classify file formats with common multi-suffix cases first."""
    name = p.name.lower()

    # Multi-suffix (most specific first)
    if name.endswith(".i3.gz"):
        return "i3.gz"
    if name.endswith(".i3.zst"):
        return "i3.zst"
    if name.endswith(".i3.bz2"):
        return "i3.bz2"

    # Single-suffix
    if name.endswith(".i3"):
        return "i3"
    if name.endswith(".zst"):
        return "zst"
    if name.endswith(".gz"):
        return "gz"
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith(".npz"):
        return "npz"
    if name.endswith(".npy"):
        return "npy"

    return "other"


def scan_dir(root: Path) -> dict:
    """Recursively scan a directory and count files by format."""
    if not root.exists():
        return {"exists": False, "total": 0, "by_format": Counter()}

    files = [p for p in root.rglob("*") if p.is_file()]
    fmt = Counter(classify_format(p) for p in files)
    return {"exists": True, "total": len(files), "by_format": fmt}


def hline(char: str = "=", n: int = 80) -> str:
    return char * n


def print_block(title: str, root: Path) -> None:
    info = scan_dir(root)

    print("\n" + hline("="))
    print(title)
    print(hline("="))
    print(f"Path           : {root}")
    print(f"Exists         : {info['exists']}")

    if not info["exists"]:
        print("Note           : Directory not found.")
        return

    print(f"Batches (files): {info['total']}")

    fmt = info["by_format"]
    if not fmt:
        print("Formats        : (no files found)")
        return

    print("\nFormats:")
    for k, v in sorted(fmt.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  - {k:<10} : {v}")

    i3_like = sum(fmt.get(x, 0) for x in ["i3", "i3.gz", "i3.zst", "i3.bz2"])
    if i3_like:
        print(f"\nI3-like total  : {i3_like}")


# =============================================================================
# 3) Dataset Definitions
# =============================================================================

@dataclass(frozen=True)
class DatasetSpec:
    title: str
    path: Path


def build_section1_datasets() -> list[DatasetSpec]:
    return [
        DatasetSpec("Muon", Path(MUON_I3PHOTONS)),
        DatasetSpec("Electron", Path(ELECTRON_I3PHOTONS)),
        DatasetSpec("Tau", Path(TAU_I3PHOTONS)),
    ]


def build_section2_datasets() -> list[DatasetSpec]:
    base_340 = Path(MAIN_340)
    return [
        DatasetSpec("Muon", base_340 / MUON_REL_340),
        DatasetSpec("Electron", base_340 / ELECTRON_REL_340),
        DatasetSpec("Tau", base_340 / TAU_REL_340),
    ]


def build_section3_datasets() -> list[DatasetSpec]:
    base_98 = Path(MAIN_98)
    return [
        DatasetSpec("Muon", base_98 / RAW_MUON_REL_98),
        DatasetSpec("Electron", base_98 / RAW_ELECTRON_REL_98),
    ]


# =============================================================================
# 3b) Summary Table Helpers
# =============================================================================

def i3_like_total_str(root: Path) -> str:
    """Return I3-like total as string; if missing -> 'MISSING'."""
    info = scan_dir(root)
    if not info["exists"]:
        return "MISSING"
    fmt = info["by_format"]
    total = sum(fmt.get(x, 0) for x in ["i3", "i3.gz", "i3.zst", "i3.bz2"])
    return str(total)


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Simple fixed-width ASCII table."""
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(r: list[str]) -> str:
        return " | ".join(r[i].ljust(widths[i]) for i in range(len(headers)))

    sep = "-+-".join("-" * w for w in widths)

    out = [fmt_row(headers), sep]
    for r in rows:
        out.append(fmt_row(r))
    return "\n".join(out)


def print_summary_table() -> None:
    # Maps: particle -> Path
    s1 = {ds.title: ds.path for ds in build_section1_datasets()}
    s2 = {ds.title: ds.path for ds in build_section2_datasets()}
    s3 = {ds.title: ds.path for ds in build_section3_datasets()}

    particles = ["Muon", "Tau", "Electron"]  # requested order

    rows: list[list[str]] = []
    for p in particles:
        c1 = i3_like_total_str(s1[p]) if p in s1 else "N/A"
        c2 = i3_like_total_str(s2[p]) if p in s2 else "N/A"
        c3 = i3_like_total_str(s3[p]) if p in s3 else "N/A"  # Tau -> N/A
        rows.append([p, c1, c2, c3])

    print("\n" + hline("#"))
    print("Summary table (I3-like totals)")
    print(hline("#"))
    print(format_table(
        ["Particle", "340 I3Photons", "340 PMT Response", "98 I3Photons"],
        rows
    ))


# =============================================================================
# 4) Main (with log-to-txt)
# =============================================================================

def run_report() -> None:
    # ----- Section 1 -----
    print("\n" + hline("#"))
    print("Section 1: 340 String - I3Photons")
    print(hline("#"))
    for ds in build_section1_datasets():
        print_block(ds.title, ds.path)

    # ----- Section 2 -----
    print("\n" + hline("#"))
    print("Section 2: 340 String - PMT Response")
    print(hline("#"))
    for ds in build_section2_datasets():
        print_block(ds.title, ds.path)

    # ----- Section 3 -----
    print("\n" + hline("#"))
    print("Section 3: 98 String - I3Photons")
    print(hline("#"))
    for ds in build_section3_datasets():
        print_block(ds.title, ds.path)

    # ----- Summary Table at the very end -----
    print_summary_table()


def main() -> None:
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = berlin_timestamp_str()
    log_path = log_dir / f"{stamp}.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee):
            run_report()
            print("\n" + hline("-"))
            print(f"Log file saved : {log_path}")
            print(hline("-"))


if __name__ == "__main__":
    main()