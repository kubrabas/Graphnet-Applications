import os 
from icecube import icetray, dataio, dataclasses, simclasses 
from collections import Counter 
import re 
from pathlib import Path 
import pandas as pd 
from icecube import dataio 
from icecube.icetray import I3Frame

IN_DIR = Path("/project/6008051/pone_simulation/MC10-000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim/Photon")
BATCH_RE = re.compile(r"cls_(\d+)\.i3(\.gz)?$")

out_csv = "/project/def-nahee/kbas/i3photons_empty_counts.csv"
out_csv_path = Path(out_csv)

# --- 1) file list ---
files = sorted([p for p in IN_DIR.glob("cls_*.i3*") if BATCH_RE.match(p.name)])
print("n_files:", len(files), "example:", files[0] if files else None)

# --- 2) RESUME: CSV varsa OK batch_id'leri oku ve skip et ---
done_ok = set()
if out_csv_path.exists() and out_csv_path.stat().st_size > 0:
    try:
        prev = pd.read_csv(out_csv_path, engine="python", on_bad_lines="skip")
        if "batch_id" in prev.columns:
            if "status" in prev.columns:
                prev_ok = prev[prev["status"] == "OK"]
            else:
                prev_ok = prev
            done_ok = set(int(x) for x in prev_ok["batch_id"].dropna().unique())
        print(f"[RESUME] OK already saved: {len(done_ok)} (FAIL will be retried)")
    except Exception as e:
        print(f"[RESUME] Could not read existing CSV, starting fresh. error={e}")

todo = []
for p in files:
    bid = int(BATCH_RE.match(p.name).group(1))
    if bid in done_ok:
        continue
    todo.append(p)

print(f"[TODO] to_process={len(todo)}")

def count_one_file(path: Path, retries: int = 2):
    m = BATCH_RE.match(path.name)
    batch_id = int(m.group(1))

    for attempt in range(retries + 1):
        n_daq = n_ph_empty = n_ph_nonempty = 0
        f = None
        try:
            f = dataio.I3File(str(path))
            while f.more():
                fr = f.pop_frame()
                if fr.Stop != I3Frame.DAQ:
                    continue
                n_daq += 1
                if (not fr.Has("I3Photons")) or (len(fr["I3Photons"]) == 0):
                    n_ph_empty += 1
                else:
                    n_ph_nonempty += 1

            return {
                "batch_id": batch_id,
                "daq": n_daq,
                "I3Photons_empty": n_ph_empty,
                "I3Photons_nonempty": n_ph_nonempty,
                "empty_frac": (n_ph_empty / n_daq) if n_daq else None,
                "status": "OK",
                "error": "",
                "path": str(path),
            }

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                continue
            return {
                "batch_id": batch_id,
                "daq": None,
                "I3Photons_empty": None,
                "I3Photons_nonempty": None,
                "empty_frac": None,
                "status": "FAIL",
                "error": err,
                "path": str(path),
            }
        finally:
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass

def append_rows(rows):
    """Append rows to CSV (create with header if not exists)."""
    df_chunk = pd.DataFrame(rows)
    write_header = (not out_csv_path.exists()) or (out_csv_path.stat().st_size == 0)
    with open(out_csv_path, "a", buffering=1) as fh:
        df_chunk.to_csv(fh, index=False, header=write_header)
        fh.flush()
        os.fsync(fh.fileno())

# --- 3) LOOP: notebook progress + every 100 flush ---
rows_buf = []
ok = 0
fail = 0

for i, p in enumerate(todo, 1):
    r = count_one_file(p, retries=2)
    rows_buf.append(r)

    if r["status"] == "OK":
        ok += 1
    else:
        fail += 1

    if i % 100 == 0:
        print(f"[{i}/{len(todo)}] OK={ok} FAIL={fail} last={p.name}")
        append_rows(rows_buf)
        rows_buf.clear()

# flush remainder
if rows_buf:
    append_rows(rows_buf)

print(f"[DONE] wrote to {out_csv}")
