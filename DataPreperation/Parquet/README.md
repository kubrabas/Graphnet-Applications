# Parquet Conversion

This directory converts PMT-response i3.gz files into GraphNeT-compatible Parquet
files, then merges them into train/val/test datasets.

## Quick Run

Submit one flavor:

```bash
python3 DataPreperation/Parquet/submit_parquet.py \
  --mc 340StringMC \
  --geometry 102_string \
  --flavor NC
```

Submit all flavors:

```bash
python3 DataPreperation/Parquet/submit_parquet.py \
  --mc 340StringMC \
  --geometry 102_string \
  --flavor all
```

`submit_parquet.py` submits one conversion job per flavor and chains one merge
job after conversion finishes.

## Output Layout

For an input PMT directory like:

```text
/home/kbas/scratch/String340MC/102_string/Electron_PMT_Response
```

conversion writes:

```text
/home/kbas/scratch/String340MC/102_string/Electron_Parquet/
  truth/
  features/
  merged_raw/
  merged/
```

Conversion and merge logs are written to:

```text
/home/kbas/scratch/String340MC/Logs/Electron_102_string_Parquet/
```

Example log names:

```text
05_06_2026_job_38837651.log
merge_Electron_102_string.out
```

## Conversion Logic

`convert_parquet.py` discovers all non-GCD I3 files under `--indir` and
processes them in parallel using GraphNeT `DataConverter` with
`num_workers=NWORKERS`. A single converter instance handles all files:

```text
PONE_Reader
I3FeatureExtractorPONE
I3TruthExtractorPONE
ParquetWriter
```

Per-file parquet outputs are written to `truth/` and `features/` under
`--outdir`. Files that already have both outputs on disk are skipped on
re-runs.

## File Handling

**DAQ-less files:** If `PONE_Reader` finds no DAQ (Q) frames in a file,
it returns an empty event list. `ParquetWriter` writes nothing for that
file. The job continues processing the remaining files.

**Pulsemap missing or empty:** Frames where the requested pulsemap key is
absent or empty are silently skipped within the file. If all frames are
filtered this way, no parquet output is written for that file.

**File open failure:** If `pop_daq()` raises an error other than
`no frame to pop`, a `RuntimeError` is raised and the job stops. Check
the SLURM log for the traceback and the offending filename.

## Job Log

The conversion log is written to:

```text
MM_DD_YYYY_job_<job_id>.log
```

It contains a job header, tqdm progress output, and a final summary:

```text
=== FINAL SUMMARY ===
skipped_previously_done : 0
input_files_discovered  : 3977
input_files_processed   : 3977
truth_outputs_found     : 3972
feature_outputs_found   : 3972
elapsed                 : 612.3s
```

`truth_outputs_found` and `feature_outputs_found` count the actual parquet
files on disk after the run. Files that were DAQ-less or fully filtered will
not appear here.

## Merge

After conversion, `merge_parquet.py` merges the per-file Parquet outputs into
batches and builds split directories:

```text
merged_raw/
merged/train/
merged/val/
merged/test/
merged/train_reindexed/
merged/val_reindexed/
merged/test_reindexed/
```

Merge logs are saved as:

```text
merge_<Flavor>_<geometry>.out
```

in the same log directory as the conversion log.

### Does GraphNeT Shuffle During Merge?

Yes. The merge step uses GraphNeT's `ParquetWriter.merge_files`.

In this repository, `merge_parquet.py` calls:

```python
writer = ParquetWriter(truth_table="truth", index_column="event_no")
writer.merge_files(
    files=[],
    output_dir=str(merged_dir),
    events_per_batch=args.events_per_batch,
    num_workers=args.num_workers,
)
```

GraphNeT's `ParquetWriter.merge_files` ignores the `files` argument and
discovers the input parquet files from the output path. It builds a master
event list from all truth parquet files, keeping both `event_no` and the source
file name. The shuffle happens in GraphNeT source code in
`ParquetWriter._identify_events`:

```python
return res.to_pandas().sample(frac=1.0)
```

That means the event list is shuffled at event level before it is split into
merged parquet batches. The shuffled list is then divided into shards of
`events_per_batch` events. For each shard, GraphNeT reads the needed source
truth/features parquet files, selects the event indices belonging to that
shard, concatenates them, and writes:

```text
merged_raw/truth/truth_<batch_id>.parquet
merged_raw/features/features_<batch_id>.parquet
```

Important detail: the GraphNeT shuffle does not pass an explicit
`random_state` to `pandas.DataFrame.sample`, so the event-level merge shuffle is
not fixed by the `seed=42` used later for train/val/test batch assignment.

### Train/Val/Test Split And Small Final Batch

After GraphNeT writes the merged batches, this repository builds splits in
`merge_parquet.py::build_splits`.

The split is done at merged-batch level, not individual-event level. The code
finds all merged truth batches:

```python
batch_ids = sorted(...)
```

Then it removes the last batch from the normal random split pool:

```python
last_batch = max(batch_ids)
main_batches = [b for b in batch_ids if b != last_batch]
```

Only `main_batches` are shuffled with the local split seed:

```python
rng = random.Random(seed)
rng.shuffle(main_batches)
```

The nominal fractions are:

```python
n_train = int(0.8 * n)
n_val = int(0.1 * n)
```

and the assignment is:

```python
train = main_batches[:n_train]
val = main_batches[n_train : n_train + n_val]
test = main_batches[n_train + n_val :] + [last_batch]
```

So the intended split is 80/10/10 by number of full merged batches, with the
last merged batch always forced into `test`.

If the total number of events is not an exact multiple of `events_per_batch`
(default: 1024 in this script), GraphNeT's last shard can contain fewer than
1024 events. Because this repository always sends `last_batch` to `test`, that
smaller remainder batch goes to the test split. It is not dropped.

The split metadata is written to:

```text
merged/split_manifest.json
```

The manifest records the seed, nominal fractions, batch counts, the selected
batch IDs, and:

```json
"last_batch_forced_to_test": <batch_id>
```
