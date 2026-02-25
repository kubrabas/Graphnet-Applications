import os
import random
import subprocess
from pathlib import Path
from typing import List, Tuple

# =========================
# Settings
# =========================
SAMPLE_SIZE_PER_DIR = 100
READ_BYTES = 1024 * 1024  # 1MB
RANDOM_SEED = 42          # Change if you want different random samples

# Only sample these file formats
ALLOWED_SUFFIXES = (".i3", ".i3.gz", ".i3.zst")

# Root folders to scan (edit as needed)
ROOT_DIRS = [
    Path("/project/6008051/pone_simulation/MC10-000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim/Photon"),
    Path("/project/6008051/pone_simulation/MC000003-nu_e-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator"),
    Path("/project/6008051/pone_simulation/MC000004-nu_tau-2_7-LeptonInjector_PROPOSAL_clsim-v10/Generator"),
    Path("/home/kbas/scratch/340_string/Muon_PMT_Response"),
    Path("/home/kbas/scratch/340_string/Electron_PMT_Response"),
    Path("/home/kbas/scratch/340_string/Tau_PMT_Response"),
    Path("/home/kbas/scratch/98_string/Muon_I3Photons"),
    Path("/home/kbas/scratch/98_string/Electron_I3Photons"),
    Path("/home/kbas/scratch/98_string/Tau_I3Photons"),
]

random.seed(RANDOM_SEED)


# =========================
# Helpers
# =========================
def is_allowed_i3_file(file_path: Path) -> bool:
    """Return True if file ends with .i3, .i3.gz, or .i3.zst (case-insensitive)."""
    name = file_path.name.lower()
    return any(name.endswith(sfx) for sfx in ALLOWED_SUFFIXES)


# =========================
# Reservoir sampling (selects random N eligible files without storing all file paths)
# =========================
def reservoir_sample_files(root_dir: Path, k: int) -> Tuple[List[Path], int]:
    sample: List[Path] = []
    total_eligible = 0

    if not root_dir.exists():
        return sample, total_eligible

    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            file_path = Path(dirpath) / name
            if not file_path.is_file():
                continue
            if not is_allowed_i3_file(file_path):
                continue

            total_eligible += 1
            if len(sample) < k:
                sample.append(file_path)
            else:
                j = random.randint(1, total_eligible)
                if j <= k:
                    sample[j - 1] = file_path

    return sample, total_eligible


# =========================
# Quick file "corruption" checks (lightweight readability tests)
# =========================
def _try_import_icecube_dataio():
    try:
        from icecube import dataio  # type: ignore
        return dataio
    except Exception:
        return None


ICECUBE_DATAIO = _try_import_icecube_dataio()


def check_file_quick(file_path: Path, read_bytes: int = READ_BYTES) -> Tuple[bool, str]:
    name = file_path.name.lower()

    try:
        # ---- IceCube I3 (plain .i3): if icecube.dataio is available, try reading 1 frame ----
        if name.endswith(".i3") and not (name.endswith(".i3.gz") or name.endswith(".i3.zst")):
            if ICECUBE_DATAIO is not None:
                f = ICECUBE_DATAIO.I3File(str(file_path))
                frame = f.pop_frame()
                f.close()
                if frame is None:
                    return False, "I3File opened but no frame (empty/truncated?)"
                return True, "OK (icecube.dataio read 1 frame)"
            else:
                # Fallback: basic binary read
                with open(file_path, "rb") as fp:
                    fp.read(read_bytes)
                return True, "OK (basic read; icecube.dataio not available)"

        # ---- gzip (.i3.gz) ----
        if name.endswith(".i3.gz"):
            import gzip
            with gzip.open(file_path, "rb") as fp:
                fp.read(read_bytes)
            return True, "OK (gzip read)"

        # ---- zstd (.i3.zst) ----
        if name.endswith(".i3.zst"):
            # First try python zstandard
            try:
                import zstandard as zstd  # type: ignore
                dctx = zstd.ZstdDecompressor()
                with open(file_path, "rb") as fh:
                    with dctx.stream_reader(fh) as reader:
                        reader.read(read_bytes)
                return True, "OK (zstd read via python zstandard)"
            except ImportError:
                # If zstandard is not installed, try `zstd -t` (integrity test)
                try:
                    proc = subprocess.run(
                        ["zstd", "-t", str(file_path)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                    )
                    if proc.returncode == 0:
                        return True, "OK (zstd -t)"
                    return False, f"zstd -t failed: {proc.stderr.strip() or proc.stdout.strip()}"
                except FileNotFoundError:
                    # If neither is available, fall back to basic read (cannot fully verify zstd integrity)
                    with open(file_path, "rb") as fp:
                        fp.read(read_bytes)
                    return True, "OK (basic read; no zstandard & no zstd binary)"

        # Should not reach here if sampling is correct, but keep a safe fallback
        with open(file_path, "rb") as fp:
            fp.read(read_bytes)
        return True, "OK (basic read)"

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# =========================
# Runner
# =========================
def main() -> None:
    for root_dir in ROOT_DIRS:
        print("=" * 80)
        print(f"ROOT: {root_dir}")
        if not root_dir.exists():
            print("  -> MISSING (directory not found)")
            continue

        sample_files, total_eligible = reservoir_sample_files(root_dir, SAMPLE_SIZE_PER_DIR)
        print(f"  Eligible files found (.i3/.i3.gz/.i3.zst): {total_eligible}")
        print(f"  Sample size                              : {len(sample_files)}")

        if total_eligible == 0:
            print("  Result                                   : No eligible files to test")
            continue

        bad: List[Tuple[Path, str]] = []
        for fp in sample_files:
            ok, msg = check_file_quick(fp)
            if not ok:
                bad.append((fp, msg))

        if not bad:
            print("  Result                                   : No corruption detected in sampled files")
        else:
            print(f"  Result                                   : {len(bad)} / {len(sample_files)} sampled files look problematic")
            for p, err in bad:
                print(f"    - {p}")
                print(f"      {err}")

    print("=" * 80)


if __name__ == "__main__":
    main()



## check the definition of "corrupt file" in this script    
## 98'de pmt responselar yok nburda glb. 
## 100 degil de hepsini mi incelesem ve txt olarak kayit mi etsem