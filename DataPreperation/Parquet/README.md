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
job after conversion finishes. If some input files fail but successful
truth/features parquet outputs exist, the merge job still runs on the available
outputs.

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

Per-file conversion logs and merge logs are written to:

```text
/home/kbas/scratch/String340MC/Logs/Electron_102_string_Parquet/
```

Example log names:

```text
electron_electron_gen_000.out
05_06_2026_job_38837651.log
merge_Electron_102_string.out
```

## Conversion Logic

`convert_parquet.py` discovers all non-GCD I3 files under `--indir`.
It processes files in parallel at the file level:

```text
ProcessPoolExecutor(max_workers=NWORKERS)
```

Inside each worker, GraphNeT `DataConverter` is run with:

```text
PONE_Reader
I3FeatureExtractorPONE
I3TruthExtractorPONE
ParquetWriter
```

`DataConverter` itself uses `num_workers=1`. This avoids nested
multiprocessing and keeps memory/I/O behavior predictable.

## Per-File Log Format

Each input file gets one log:

```text
<stem>.out
```

Normal successful example:

```text
[electron_electron_gen_000.i3.gz] kept=12  noise_only=0  pulsemap_does_not_exist=0
=== SUCCESS  elapsed=10.2s ===
category : successfully_transferred_kept_events
@ 2026-05-06 11:10:48 (Berlin)
```

Completely empty example:

```text
[electron_electron_gen_1080.i3.gz] kept=0  noise_only=0  pulsemap_does_not_exist=0
=== SUCCESS  elapsed=0.1s ===
category : completely_empty_file
@ 2026-05-06 11:10:48 (Berlin)
```

File-open failure example:

```text
[FILE ERROR] Could not open: /path/to/file.i3.gz  error=...
=== FAILED  elapsed=0.1s ===
category : failed_to_open_file
```

## Counters

`kept`
: Number of events that passed filtering and were handed to the extractors.

`noise_only`
: Number of frames where the requested pulsemap exists but is empty.

`pulsemap_does_not_exist`
: Number of frames where the requested pulsemap key is missing.

Frame read failures are not counted as `corrupt_frames`. If `pop_daq()` fails,
the file fails immediately. The log still prints the counters collected before
the failure.

## File Categories

Every processed file is assigned exactly one category.

`successfully_transferred_kept_events`
: Conversion succeeded and `kept > 0`. Expected truth/features Parquet files
exist.

`completely_empty_file`
: Conversion succeeded with `kept=0`, `noise_only=0`, and
`pulsemap_does_not_exist=0`. This usually means no usable DAQ/event frames
survived to this stage.

`only_noise_events`
: Conversion succeeded with `kept=0`, `noise_only > 0`, and
`pulsemap_does_not_exist=0`.

`only_missing_pulsemap_events`
: Conversion succeeded with `kept=0`, `noise_only=0`, and
`pulsemap_does_not_exist > 0`.

`only_filtered_events`
: Conversion succeeded with `kept=0`, `noise_only > 0`, and
`pulsemap_does_not_exist > 0`.

`failed_to_open_file`
: The I3 file could not be opened.

`failed_at_first_event`
: The file opened, but conversion failed before any event was kept or filtered.

`failed_after_partial_progress`
: Conversion failed after at least one event was kept.

`failed_after_only_filtered_events`
: Conversion failed after seeing only filtered events, such as noise-only or
missing-pulsemap frames.

`failed_missing_parquet_outputs_after_success`
: The reader reported kept events, but the expected truth/features Parquet
outputs were not present.

## General Job Log

The general conversion log is written in the same log directory as the per-file
logs:

```text
MM_DD_YYYY_job_<job_id>.log
```

Each processed file is logged with category and counters:

```text
[12:34:56] [successfully_transferred_kept_events] electron_electron_gen_000.i3.gz  kept=12 noise_only=0 pulsemap_does_not_exist=0  (10.2s)
```

The final summary includes validation checks:

```text
=== FINAL SUMMARY ===
successfully_transferred_kept_events : 3972
completely_empty_file : 5
only_noise_events : 0
only_missing_pulsemap_events : 0
only_filtered_events : 0
failed_to_open_file : 0
failed_at_first_event : 0
failed_after_partial_progress : 0
failed_after_only_filtered_events : 0
failed_missing_parquet_outputs_after_success : 0
category_total : 3977
failed_total : 0
skipped_previously_done : 0
input_files_discovered : 3977
per_file_logs_found : 3977
accounting_check : PASS
per_file_log_check : PASS
```

For a fresh run, the important lines are:

```text
failed_total : 0
accounting_check : PASS
per_file_log_check : PASS
```

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

in the same log directory as the per-file conversion logs.

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
