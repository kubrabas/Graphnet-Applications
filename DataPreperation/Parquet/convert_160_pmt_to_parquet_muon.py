from __future__ import annotations

# ============================================================
# Imports
# ============================================================

import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

from graphnet.utilities.imports import has_icecube_package

if has_icecube_package():
    from icecube import icetray  # noqa: F401
else:
    raise RuntimeError("IceCube/IceTray environment not available.")

# ============================================================
# Repo path setup (to import MyClasses/*)
# Script assumed at:
#   Graphnet-Applications/DataPreperation/Parquet/convert_160_pmt_to_parquet_muon.py
# so repo root is parents[2]
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

# Custom classes
from MyClasses.reader import PONE_Reader  # noqa: E402
from MyClasses.feature_extractor import I3FeatureExtractorPONE  # noqa: E402
from MyClasses.truth_extractor import I3TruthExtractorPONE  # noqa: E402

# Filters
from graphnet.data.extractors.icecube.utilities.i3_filters import (  # noqa: E402
    I3Filter,
    NullSplitI3Filter,
)

# DataConverter + Writer
from graphnet.data.dataconverter import DataConverter  # noqa: E402
from graphnet.data.writers import ParquetWriter  # noqa: E402


# ============================================================
# I3 filters
# ============================================================

class NonEmptyPulseSeriesI3Filter(I3Filter):
    """Drop frame if given PulseSeriesMap/Mask is empty (has 0 pulses)."""

    def __init__(self, pulsemap_name: str = "EventPulseSeries_nonoise"):
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._pulsemap_name = pulsemap_name

    def _keep_frame(self, frame: "icetray.I3Frame") -> bool:
        if not frame.Has(self._pulsemap_name):
            return False

        pm = frame[self._pulsemap_name]

        # Handle MapMask vs Map
        try:
            if hasattr(pm, "apply"):
                pm = pm.apply(frame)
        except Exception:
            # If apply fails for any reason, do not keep the frame
            return False

        # Count total pulses across all OMKeys/PMTs
        try:
            total = 0
            for _, series in pm.items():
                total += len(series)
                if total > 0:
                    return True
            return False
        except Exception:
            # Fallback: if it behaves like a container
            try:
                return len(pm) > 0
            except Exception:
                return False


# ============================================================
# Paths / config
# ============================================================

INPUT_GLOB = "/scratch/kbas/160_string/Muon_PMT_Response/*.i3.gz"
INPUT_ROOT = "/scratch/kbas/160_string/Muon_PMT_Response/"
OUTDIR = "/scratch/kbas/FinalParquetDatasets/160_string_muon_nonoise"
GCD_RESCUE = "/scratch/kbas/160_string/GCD_160strings.i3.gz"


def batch_id_from_i3(path):
    m = re.search(r"muon_batch_(\d+)\.i3\.gz$", os.path.basename(path))
    return int(m.group(1)) if m else None


def batch_ids_in_outdir(outdir):
    # Scan anything containing "muon_batch_<id>" under outdir
    candidates = glob.glob(os.path.join(outdir, "**", "*"), recursive=True)
    ids = set()
    for p in candidates:
        m = re.search(r"muon_batch_(\d+)", os.path.basename(p))
        if m:
            ids.add(int(m.group(1)))
    return ids


# ============================================================
# Discover inputs / already-processed batches
# ============================================================

all_files = sorted(glob.glob(INPUT_GLOB))
done_ids = batch_ids_in_outdir(OUTDIR)


# ============================================================
# Reader, extractors, writer, converter
# ============================================================

reader = PONE_Reader(
    gcd_rescue=GCD_RESCUE,
    i3_filters=[NullSplitI3Filter(), NonEmptyPulseSeriesI3Filter("EventPulseSeries_nonoise")],
)

extractors = [
    I3FeatureExtractorPONE(
        pulsemap="EventPulseSeries_nonoise",
        name="features",
        exclude=[
            "pmt_area",
            "rde",
            "width",
            "event_time",
            "is_bright_dom",
            "is_saturated_dom",
            "is_errata_dom",
            "is_bad_dom",
            "hlc",
            "awtd",
            "dom_type",
        ],
    ),
    I3TruthExtractorPONE(
        mctree="I3MCTree_postprop",
        name="truth",
        exclude=[
            "L7_oscNext_bool",
            "L6_oscNext_bool",
            "L5_oscNext_bool",
            "L4_oscNext_bool",
            "L3_oscNext_bool",
            "OnlineL2Filter_17",
            "MuonFilter_13",
            "CascadeFilter_13",
            "DeepCoreFilter_13",
            "dbang_decay_length",
            "inelasticity",
        ],
    ),
]

writer = ParquetWriter(truth_table="truth", index_column="event_no")

converter = DataConverter(
    file_reader=reader,
    save_method=writer,
    extractors=extractors,
    outdir=OUTDIR,
    num_workers=16,
    index_column="event_no",
)


# ============================================================
# Convert I3 -> Parquet
# ============================================================

converter(input_dir=INPUT_ROOT)

print("DONE:", OUTDIR)


# ============================================================
# Merge batches
# ============================================================

MERGED_DIR = os.path.join(OUTDIR, "merged")

writer.merge_files(
    files=[],
    output_dir=MERGED_DIR,
    events_per_batch=1024,
    num_workers=16,
)


# ============================================================
# Rename merged -> merged_raw
# ============================================================

MERGED_DIR = Path(MERGED_DIR)
MERGED_RAW = MERGED_DIR.parent / "merged_raw"

# Move merged -> merged_raw (if not already done)
if not MERGED_RAW.exists():
    os.rename(MERGED_DIR, MERGED_RAW)
    print("moved:", MERGED_DIR, "->", MERGED_RAW)
else:
    print("merged_raw already exists:", MERGED_RAW)


# ============================================================
# Build train/val/test split over merged batches
# ============================================================

truth_dir = MERGED_RAW / "truth"
feat_dir = MERGED_RAW / "features"

pat = re.compile(r"^truth_(\d+)\.parquet$")
batch_ids = sorted(
    int(pat.match(p.name).group(1))
    for p in truth_dir.glob("truth_*.parquet")
    if pat.match(p.name)
)

last_batch = max(batch_ids)
main_batches = [b for b in batch_ids if b != last_batch]

seed = 42
import random  # keep local usage identical to original intent
rng = random.Random(seed)
rng.shuffle(main_batches)

n = len(main_batches)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

splits = {
    "train": main_batches[:n_train],
    "val": main_batches[n_train : n_train + n_val],
    "test": main_batches[n_train + n_val :] + [last_batch],
}


# ============================================================
# Materialize split directories (hardlink -> symlink -> copy)
# ============================================================

def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hardlink preferred
    except Exception:
        try:
            os.symlink(src, dst)  # fallback symlink
        except Exception:
            shutil.copy2(src, dst)  # fallback copy


NEW_MERGED = MERGED_RAW.parent / "merged"
tables = ["truth", "features"]

for split_name, ids in splits.items():
    for table in tables:
        for bid in ids:
            src = (MERGED_RAW / table / f"{table}_{bid}.parquet")
            dst = (NEW_MERGED / split_name / table / src.name)
            if src.exists():
                link_or_copy(src, dst)

manifest = {
    "seed": seed,
    "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
    "last_batch_forced_to_test": int(last_batch),
    "counts_in_batches": {k: len(v) for k, v in splits.items()},
    "splits": splits,
}

with open(NEW_MERGED / "split_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("done:", NEW_MERGED)
print({k: len(v) for k, v in splits.items()})


# ============================================================
# Reindex splits (create *_reindexed directories using symlinks)
# ============================================================

def reindex_split(old_split_dir: str, new_split_dir: str):
    old = Path(old_split_dir)
    new = Path(new_split_dir)
    (new / "truth").mkdir(parents=True, exist_ok=True)
    (new / "features").mkdir(parents=True, exist_ok=True)

    rx = re.compile(r"_(\d+)\.parquet$")
    ids = sorted(
        int(rx.search(p.name).group(1))
        for p in (old / "truth").glob("truth_*.parquet")
    )

    for new_id, old_id in enumerate(ids):
        for table in ["truth", "features"]:
            src = old / table / f"{table}_{old_id}.parquet"
            dst = new / table / f"{table}_{new_id}.parquet"
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)


for split_name in ["train", "val", "test"]:
    reindex_split(
        str(NEW_MERGED / split_name),
        str(NEW_MERGED / f"{split_name}_reindexed"),
    )