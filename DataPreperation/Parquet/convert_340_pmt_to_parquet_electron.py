import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# ----------------------------
# Make sure we can import MyClasses/*
# Script is assumed at:
#   Graphnet-Applications/DataPreperation/Parquet/convert_340_pmt_to_parquet_electron.py
# so repo root is parents[2]
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

# Your custom classes
from MyClasses.reader import PONE_Reader  # noqa
from MyClasses.feature_extractor import I3FeatureExtractorPONE  # noqa
from MyClasses.truth_extractor import I3TruthExtractorPONE  # noqa

# IceTray
from graphnet.utilities.imports import has_icecube_package
if not has_icecube_package():
    raise RuntimeError("IceCube/IceTray environment not available (has_icecube_package() == False).")

from icecube import dataio, icetray  # noqa: E402

# Arrow parquet writer (streaming-friendly)
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


BATCH_RE = re.compile(r"batch_(\d+)\.i3(?:\.gz)?$")


def extract_batch_id(path: str) -> Optional[int]:
    m = BATCH_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def list_sorted_i3_files(input_dir: str, pattern: str) -> List[str]:
    files = glob.glob(os.path.join(input_dir, pattern))
    # fallback: include any *.i3.gz if pattern is too strict
    if not files:
        files = glob.glob(os.path.join(input_dir, "*.i3.gz"))

    def key_fn(p: str):
        bid = extract_batch_id(p)
        # Sort: (has_bid, bid, filename)
        # has_bid=0 first if bid exists, else 1
        return (0, bid, os.path.basename(p)) if bid is not None else (1, 10**18, os.path.basename(p))

    files = sorted(files, key=key_fn)
    return files


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n")


def count_ok(status_dir: Path) -> int:
    return len(list(status_dir.glob("ok_batch_*.txt")))


def parse_csv_list(x: str) -> List[str]:
    x = x.strip()
    if not x:
        return []
    return [t.strip() for t in x.split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-index", type=int, required=True, help="SLURM array index (0-based).")
    ap.add_argument("--gcd", type=str, required=True, help="GCD rescue path.")
    ap.add_argument("--input-dir", type=str, required=True, help="Directory containing Electron_PMT_Response i3.gz batches.")
    ap.add_argument("--output-dir", type=str, required=True, help="Output dataset root (will create truth/ + features/).")
    ap.add_argument("--pulsemap", type=str, default="EventPulseSeries_nonoise", help="Pulse series name.")
    ap.add_argument("--mctree", type=str, default="I3MCTree_postprop", help="MCTree name.")
    ap.add_argument("--pattern", type=str, default="electron_batch_*.i3.gz", help="Glob pattern inside input-dir.")
    ap.add_argument("--event-no-stride", type=int, default=1_000_000, help="event_no = batch_id*stride + local_event_index")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing parquet outputs.")
    ap.add_argument("--features-exclude", type=str, default="pmt_area,rde,width,event_time,is_bright_dom,is_saturated_dom,is_errata_dom,is_bad_dom,hlc,awtd,dom_type",
                    help="Comma-separated features to exclude (passed into extractor).")
    ap.add_argument("--truth-exclude", type=str, default="L7_oscNext_bool,L6_oscNext_bool,L5_oscNext_bool,L4_oscNext_bool,L3_oscNext_bool,OnlineL2Filter_17,MuonFilter_13,CascadeFilter_13,DeepCoreFilter_13,dbang_decay_length,inelasticity",
                    help="Comma-separated truth keys to exclude (passed into extractor).")
    ap.add_argument("--flush-pulses", type=int, default=500_000, help="Flush features parquet every N pulses (approx).")

    args = ap.parse_args()

    t0 = time.time()

    in_files = list_sorted_i3_files(args.input_dir, args.pattern)
    n_total = len(in_files)

    print(f"[INFO] input-dir     = {args.input_dir}")
    print(f"[INFO] pattern       = {args.pattern}")
    print(f"[INFO] total files    = {n_total}")
    print(f"[INFO] file-index     = {args.file_index}")

    if n_total == 0:
        raise RuntimeError("No input .i3.gz files found.")

    if args.file_index < 0 or args.file_index >= n_total:
        print(f"[SKIP] file-index {args.file_index} out of range (0..{n_total-1}). Exiting cleanly.")
        return

    i3_path = in_files[args.file_index]
    batch_id = extract_batch_id(i3_path)
    if batch_id is None:
        # fallback if naming deviates
        batch_id = args.file_index

    out_root = Path(args.output_dir)
    feat_dir = out_root / "features"
    truth_dir = out_root / "truth"
    status_dir = out_root / "_status"

    feat_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    feat_path = feat_dir / f"features_{batch_id}.parquet"
    truth_path = truth_dir / f"truth_{batch_id}.parquet"

    ok_marker = status_dir / f"ok_batch_{batch_id}.txt"
    fail_marker = status_dir / f"fail_batch_{batch_id}.txt"

    print(f"[INFO] selected file  = {i3_path}")
    print(f"[INFO] batch_id       = {batch_id}")
    print(f"[INFO] output features= {feat_path}")
    print(f"[INFO] output truth   = {truth_path}")

    if (feat_path.exists() or truth_path.exists()) and (not args.overwrite):
        print("[SKIP] Output parquet already exists (use --overwrite to force).")
        # still mark ok, çünkü zaten done sayılabilir
        if not ok_marker.exists():
            touch(ok_marker)
        done = count_ok(status_dir)
        print(f"[PROGRESS] ok={done}/{n_total} (status markers)")
        return

    # Build extractors (your custom ones)
    features_exclude = parse_csv_list(args.features_exclude)
    truth_exclude = parse_csv_list(args.truth_exclude)

    feature_extractor = I3FeatureExtractorPONE(
        pulsemap=args.pulsemap,
        name="features",
        exclude=features_exclude,
    )
    truth_extractor = I3TruthExtractorPONE(
        mctree=args.mctree,
        name="truth",
        exclude=truth_exclude,
    )

    # Minimal reader just to reuse your skip logic for pulsemap existence/emptiness
    # (We don't call reader(file) to avoid holding everything in memory.)
    reader = PONE_Reader(
        gcd_rescue=args.gcd,
        i3_filters=None,
        pulsemap=args.pulsemap,
        skip_empty_pulses=True,
    )

    # Set GCD once for both extractors
    feature_extractor.set_gcd(i3_file=i3_path, gcd_file=args.gcd)
    truth_extractor.set_gcd(i3_file=i3_path, gcd_file=args.gcd)

    # Temp output for atomic replace
    feat_tmp = feat_path.with_name(feat_path.name + ".tmp")
    truth_tmp = truth_path.with_name(truth_path.name + ".tmp")

    # Remove old tmp if any
    if feat_tmp.exists():
        feat_tmp.unlink()
    if truth_tmp.exists():
        truth_tmp.unlink()

    # Streaming parquet writer for features
    feat_writer: Optional[pq.ParquetWriter] = None
    feat_buf: Dict[str, List[Any]] = {}
    feat_buf_pulses = 0

    truth_rows: List[Dict[str, Any]] = []

    n_events = 0
    n_pulses_total = 0
    local_event_index = 0

    def flush_features():
        nonlocal feat_writer, feat_buf, feat_buf_pulses
        if feat_buf_pulses == 0:
            return
        table = pa.Table.from_pydict(feat_buf)
        if feat_writer is None:
            feat_writer = pq.ParquetWriter(str(feat_tmp), table.schema)
        feat_writer.write_table(table)
        # reset
        feat_buf = {}
        feat_buf_pulses = 0

    try:
        i3 = dataio.I3File(i3_path, "r")

        while i3.more():
            try:
                frame = i3.pop_frame()
            except Exception as e:
                # corrupted frame etc.
                if "I3" in str(e):
                    continue
                raise

            # We accept any frame that passes your reader's skip logic
            # (mainly: pulsemap exists & non-empty)
            try:
                if reader._skip_frame(frame):  # uses pulsemap checks
                    continue
            except Exception:
                # if skip logic fails for some reason, be conservative
                continue

            # Extract
            feats = feature_extractor(frame)   # dict of lists
            truth = truth_extractor(frame)     # dict of scalars

            # Determine number of pulses (rows)
            # use "charge" if exists, else first list-like
            if "charge" in feats:
                n_pulses = len(feats["charge"])
            else:
                # fallback: find first list value
                n_pulses = 0
                for v in feats.values():
                    if isinstance(v, list):
                        n_pulses = len(v)
                        break

            if n_pulses == 0:
                continue

            event_no = batch_id * args.event_no_stride + local_event_index
            local_event_index += 1

            # append truth row
            truth_row = dict(truth)
            truth_row["event_no"] = int(event_no)
            truth_row["batch_id"] = int(batch_id)
            truth_rows.append(truth_row)

            # append features rows into buffer
            # ensure all feature columns exist in buffer
            for k, v in feats.items():
                if k not in feat_buf:
                    feat_buf[k] = []
                feat_buf[k].extend(v)

            # event_no column (repeat per pulse)
            if "event_no" not in feat_buf:
                feat_buf["event_no"] = []
            feat_buf["event_no"].extend([int(event_no)] * n_pulses)

            if "batch_id" not in feat_buf:
                feat_buf["batch_id"] = []
            feat_buf["batch_id"].extend([int(batch_id)] * n_pulses)

            feat_buf_pulses += n_pulses
            n_events += 1
            n_pulses_total += n_pulses

            if feat_buf_pulses >= args.flush_pulses:
                flush_features()

        # final flush + close
        flush_features()
        if feat_writer is not None:
            feat_writer.close()

        # write truth
        truth_table = pa.Table.from_pylist(truth_rows)
        pq.write_table(truth_table, str(truth_tmp))

        # atomic replace
        os.replace(str(feat_tmp), str(feat_path))
        os.replace(str(truth_tmp), str(truth_path))

        # mark ok
        if fail_marker.exists():
            fail_marker.unlink()
        touch(ok_marker)

        done = count_ok(status_dir)
        dt = time.time() - t0
        print(f"[DONE] events={n_events} pulses={n_pulses_total} time={dt/60:.2f} min")
        print(f"[PROGRESS] ok={done}/{n_total} (status markers)")

    except Exception as e:
        print(f"[FAIL] batch_id={batch_id} file={i3_path}")
        print(f"[FAIL] error: {repr(e)}")
        # cleanup tmp
        try:
            if feat_writer is not None:
                feat_writer.close()
        except Exception:
            pass
        if feat_tmp.exists():
            feat_tmp.unlink()
        if truth_tmp.exists():
            truth_tmp.unlink()
        fail_marker.write_text(f"{repr(e)}\n")
        raise


if __name__ == "__main__":
    main()