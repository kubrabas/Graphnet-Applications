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
# Configuration
# =============================================================================

OUTPUT_LOG_DIR = Path("/project/def-nahee/kbas/Graphnet-Applications/Playground/Datasets")
TIMEZONE_NAME = "Europe/Berlin"

I3_LIKE_EXTENSIONS = ("i3", "i3.gz", "i3.zst", "i3.bz2")


# =============================================================================
# Dataset Paths
# =============================================================================

# ---- Section 1: 340 String - I3Photons (absolute locations) ----
I3PHOTONS_340_PATHS = {
    "Muon": Path("/project/6008051/pone_simulation/MC10-000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim/Photon"),
    "Electron": Path("/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator"),
    "Tau": Path("/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator"),
}

# ---- Section 2: 340 String - PMT Response ----
PMT_RESPONSE_340_BASE = Path("/home/kbas/scratch/340_string")
PMT_RESPONSE_340_SUBDIRS = {
    "Muon": "Muon_PMT_Response",
    "Electron": "Electron_PMT_Response",
    "Tau": "Tau_PMT_Response",
}

# ---- Section 3: 102 String - I3Photons ----
I3PHOTONS_102_BASE = Path("/home/kbas/scratch/102_string")
I3PHOTONS_102_SUBDIRS = {
    "Muon": "Muon_I3Photons",
    "Electron": "Electron_I3Photons",
    "Tau": "Tau_I3Photons",
}

# ---- Section 4: 102 String - PMT Response ----
PMT_RESPONSE_102_BASE = Path("/home/kbas/scratch/102_string")
PMT_RESPONSE_102_SUBDIRS = {
    "Muon": "Muon_PMT_Response",
    "Electron": "Electron_PMT_Response",
    "Tau": "Tau_PMT_Response",
}

# ---- Section 5: 160 String - I3Photons (NEW, same as 102) ----
I3PHOTONS_160_BASE = Path("/home/kbas/scratch/160_string")
I3PHOTONS_160_SUBDIRS = {
    "Muon": "Muon_I3Photons",
    "Electron": "Electron_I3Photons",
    "Tau": "Tau_I3Photons",
}

# ---- Section 6: 160 String - PMT Response (NEW, same as 102) ----
PMT_RESPONSE_160_BASE = Path("/home/kbas/scratch/160_string")
PMT_RESPONSE_160_SUBDIRS = {
    "Muon": "Muon_PMT_Response",
    "Electron": "Electron_PMT_Response",
    "Tau": "Tau_PMT_Response",
}


# =============================================================================
# Utilities
# =============================================================================

def berlin_timestamp_str() -> str:
    if ZoneInfo is not None:
        tz = ZoneInfo(TIMEZONE_NAME)
        now = datetime.now(tz)
    else:
        now = datetime.now()
    return now.strftime("%d_%m_%Y_%H_%M_%S")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def separator(char: str = "=", width: int = 80) -> str:
    return char * width


def detect_file_format(path: Path) -> str:
    name = path.name.lower()

    if name.endswith(".i3.gz"):
        return "i3.gz"
    if name.endswith(".i3.zst"):
        return "i3.zst"
    if name.endswith(".i3.bz2"):
        return "i3.bz2"

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


@dataclass(frozen=True)
class DirectoryScanResult:
    exists: bool
    total_files: int
    by_format: Counter
    i3_like_total: int


def scan_directory(root_dir: Path) -> DirectoryScanResult:
    if not root_dir.exists():
        return DirectoryScanResult(False, 0, Counter(), 0)

    try:
        files = [p for p in root_dir.rglob("*") if p.is_file()]
    except Exception:
        return DirectoryScanResult(True, 0, Counter(), 0)

    format_counts = Counter(detect_file_format(p) for p in files)
    i3_like_total = sum(format_counts.get(ext, 0) for ext in I3_LIKE_EXTENSIONS)

    return DirectoryScanResult(True, len(files), format_counts, i3_like_total)


def print_directory_report(title: str, root_dir: Path) -> None:
    result = scan_directory(root_dir)

    print("\n" + separator("="))
    print(title)
    print(separator("="))
    print(f"Path           : {root_dir}")
    print(f"Exists         : {result.exists}")

    if not result.exists:
        print("Note           : Directory not found.")
        return

    print(f"Batches (files): {result.total_files}")

    if not result.by_format:
        print("Formats        : (no files found)")
        return

    print("\nFormats:")
    for fmt, count in sorted(result.by_format.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  - {fmt:<10} : {count}")

    if result.i3_like_total:
        print(f"\nI3-like total  : {result.i3_like_total}")


# =============================================================================
# Builders
# =============================================================================

@dataclass(frozen=True)
class DatasetSpec:
    label: str
    path: Path


def build_specs_from_mapping(mapping: dict[str, Path]) -> list[DatasetSpec]:
    return [DatasetSpec(label=particle, path=path) for particle, path in mapping.items()]


def build_specs_from_base(base_dir: Path, subdirs: dict[str, str]) -> list[DatasetSpec]:
    return [DatasetSpec(label=particle, path=base_dir / subdir) for particle, subdir in subdirs.items()]


def build_section1_specs(): return build_specs_from_mapping(I3PHOTONS_340_PATHS)
def build_section2_specs(): return build_specs_from_base(PMT_RESPONSE_340_BASE, PMT_RESPONSE_340_SUBDIRS)
def build_section3_specs(): return build_specs_from_base(I3PHOTONS_102_BASE, I3PHOTONS_102_SUBDIRS)
def build_section4_specs(): return build_specs_from_base(PMT_RESPONSE_102_BASE, PMT_RESPONSE_102_SUBDIRS)
def build_section5_specs(): return build_specs_from_base(I3PHOTONS_160_BASE, I3PHOTONS_160_SUBDIRS)
def build_section6_specs(): return build_specs_from_base(PMT_RESPONSE_160_BASE, PMT_RESPONSE_160_SUBDIRS)


# =============================================================================
# Summary Table
# =============================================================================

def i3_like_total_str(root_dir: Path) -> str:
    result = scan_directory(root_dir)
    if not result.exists:
        return "MISSING"
    return str(result.i3_like_total)


def format_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row):
        return " | ".join(row[i].ljust(widths[i]) for i in range(len(headers)))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt_row(headers), sep]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


def print_summary_table():
    s1 = {ds.label: ds.path for ds in build_section1_specs()}
    s2 = {ds.label: ds.path for ds in build_section2_specs()}
    s3 = {ds.label: ds.path for ds in build_section3_specs()}
    s4 = {ds.label: ds.path for ds in build_section4_specs()}
    s5 = {ds.label: ds.path for ds in build_section5_specs()}
    s6 = {ds.label: ds.path for ds in build_section6_specs()}

    particle_order = ["Muon", "Tau", "Electron"]

    rows = []
    for p in particle_order:
        rows.append([
            p,
            i3_like_total_str(s1[p]),
            i3_like_total_str(s2[p]),
            i3_like_total_str(s3[p]),
            i3_like_total_str(s4[p]),
            i3_like_total_str(s5[p]),
            i3_like_total_str(s6[p]),
        ])

    print("\n" + separator("#"))
    print("Summary table (I3-like totals)")
    print(separator("#"))
    print(format_table(
        ["Particle", "340 I3Photons", "340 PMT", "102 I3Photons", "102 PMT", "160 I3Photons", "160 PMT"],
        rows
    ))


# =============================================================================
# Main
# =============================================================================

def run_report():
    print("\n" + separator("#"))
    print("Section 1: 340 String - I3Photons")
    print(separator("#"))
    for ds in build_section1_specs():
        print_directory_report(ds.label, ds.path)

    print("\n" + separator("#"))
    print("Section 2: 340 String - PMT Response")
    print(separator("#"))
    for ds in build_section2_specs():
        print_directory_report(ds.label, ds.path)

    print("\n" + separator("#"))
    print("Section 3: 102 String - I3Photons")
    print(separator("#"))
    for ds in build_section3_specs():
        print_directory_report(ds.label, ds.path)

    print("\n" + separator("#"))
    print("Section 4: 102 String - PMT Response")
    print(separator("#"))
    for ds in build_section4_specs():
        print_directory_report(ds.label, ds.path)

    print("\n" + separator("#"))
    print("Section 5: 160 String - I3Photons")
    print(separator("#"))
    for ds in build_section5_specs():
        print_directory_report(ds.label, ds.path)

    print("\n" + separator("#"))
    print("Section 6: 160 String - PMT Response")
    print(separator("#"))
    for ds in build_section6_specs():
        print_directory_report(ds.label, ds.path)

    print_summary_table()


def main():
    OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    stamp = berlin_timestamp_str()
    log_file = OUTPUT_LOG_DIR / f"{stamp}.txt"

    with open(log_file, "w", encoding="utf-8") as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee):
            run_report()
            print("\n" + separator("-"))
            print(f"Log file saved : {log_file}")
            print(separator("-"))


if __name__ == "__main__":
    main()