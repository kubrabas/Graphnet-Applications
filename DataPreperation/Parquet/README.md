# Parquet Conversion

This directory converts PMT-response `.i3` files into GraphNeT-compatible
Parquet datasets. The default submission workflow now builds the three standard
geometries, merges each dataset into train/val/test splits, writes RobustScaler
percentile CSVs, and builds categorized Parquet splits.

## Quick Run

Submit one flavor for all default geometries:

```bash
python3 DataPreperation/Parquet/submit_parquet.py \
  --mc 340StringMC \
  --flavor Electron
```

Submit all flavors for all default geometries:

```bash
python3 DataPreperation/Parquet/submit_parquet.py \
  --mc 340StringMC \
  --flavor all
```

By default, omitting `--geometry` runs:

```text
full_geometry
160_string
102_string
```

You can still submit only one geometry, for example:

```bash
python3 DataPreperation/Parquet/submit_parquet.py \
  --mc 340StringMC \
  --geometry 102_string \
  --flavor Electron
```

Check the planned jobs without submitting them:

```bash
python3 DataPreperation/Parquet/submit_parquet.py \
  --dry-run \
  --mc 340StringMC \
  --flavor Electron
```

## Default Workflow

For each selected geometry/flavor, `submit_parquet.py` does all of this:

1. Submits one conversion job running `convert_parquet.py`.
2. Chains one merge job after conversion succeeds.
3. Chains categorized Parquet jobs after merge succeeds for the default
   category columns.

Merge and categorized outputs are not optional in this workflow. There is no
`--with-merge` or `--with-categories` flag.

The default category columns are:

```text
category1_isMuonCC
category2_tauCC_others_muonCC
category_3_contains_muon
```

## Geometry Filters

Each geometry is filtered independently with its own nonoise trigger flag:

```text
full_geometry -> triggered_nonoise_340_string == 1
160_string    -> triggered_nonoise_160_string == 1
102_string    -> triggered_nonoise_102_string == 1
```

The pulsemap used for each geometry is:

```text
full_geometry -> PMT_Response_nonoise_340_String
160_string    -> PMT_Response_nonoise_160_String
102_string    -> PMT_Response_nonoise_102_String
```

`340_string` is also understood by `convert_parquet.py` and maps to
`triggered_nonoise_340_string`, but it is not part of the default submit list.

The old `common_events` behavior has been removed. The pipeline no longer
builds `features_340/`, `features_160/`, or `features_102/` inside one dataset.
Each geometry writes one normal `features/` table.

## Energy Filter

The default max-energy filter is enabled:

```text
EventProperties.totalEnergy <= 1e6 GeV
```

This adds the suffix `Emax1e6` to output and log directories. For example:

```text
Electron_Parquet_Emax1e6
Electron_102_string_Parquet_Emax1e6
```

## Output Layout

For `340StringMC`, `Electron`, and `102_string`, conversion writes raw per-file
Parquet outputs under:

```text
/home/kbas/scratch/String340MC_pone_offline_version3_plus/Parquet/102_string/Electron_Parquet_Emax1e6/
  truth/
  features/
```

The merge job then writes:

```text
  merged_raw/
  merged/train/
  merged/val/
  merged/test/
  merged/train_reindexed/
  merged/val_reindexed/
  merged/test_reindexed/
  merged/split_manifest.json
```

The categorized jobs write one dataset per category column/value:

```text
  categorized/<category_column>/category<value>/train/
  categorized/<category_column>/category<value>/val/
  categorized/<category_column>/category<value>/test/
  categorized/<category_column>/category<value>/split_manifest.json
```

For `full_geometry`, the output folder uses `Full_Geometry`:

```text
/home/kbas/scratch/String340MC_pone_offline_version3_plus/Parquet/Full_Geometry/Electron_Parquet_Emax1e6/
```

Logs are written to:

```text
/home/kbas/scratch/String340MC_pone_offline_version3_plus/Logs/<Flavor>_<geometry>_Parquet_Emax1e6/
```

Example log names:

```text
MM_DD_YYYY_job_<job_id>.log
merge_<Flavor>_<geometry>.out
categorization_<category_column>_<job_id>.log
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

The conversion filters are:

```text
NullSplitI3Filter
TriggeredNonoiseFilter for the selected geometry
EventPropertiesMaxEnergyFilter, unless max energy is disabled
```

Per-file parquet outputs are written to `truth/` and `features/` under
`--outdir`. Files that already have both outputs on disk are skipped on re-runs.

## Truth Labels

`I3TruthExtractorPONE` writes these category columns to the truth table:

```text
category1_isMuonCC
  1 : NuMu/NuMuBar charged-current event
  0 : all other events

category2_tauCC_others_muonCC
  0  : tau_CC
  1  : all NC + electron CC
  2  : muon_CC
  -1 : padding / not assigned

category_3_contains_muon
  1 : post-propagation event contains a muon
  0 : no post-propagation muon found
```

`category1_isMuonCC` is based on the parent neutrino and charged-current flag.
`category_3_contains_muon` checks whether the post-propagation particle tree
contains a muon anywhere. These are intentionally different labels.

## File Handling

**DAQ-less files:** If `PONE_Reader` finds no DAQ frames in a file, it returns
an empty event list. `ParquetWriter` writes nothing for that file. The job
continues processing the remaining files.

**Pulsemap missing or empty:** Frames where the requested pulsemap key is absent
or empty are skipped by the reader. If all frames are filtered this way, no
parquet output is written for that file.

**Missing trigger flag:** `TriggeredNonoiseFilter` raises a `RuntimeError` if the
required trigger flag is missing from a frame.

**File open failure:** If `pop_daq()` raises an error other than `no frame to
pop`, a `RuntimeError` is raised and the job stops. Check the SLURM log for the
traceback and the offending filename.

## Conversion Log

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
feature_outputs_found   : {'features': 3972}
elapsed                 : 612.3s
```

`truth_outputs_found` and `feature_outputs_found` count the actual parquet files
on disk after the run. Files that were DAQ-less or fully filtered will not
appear here.

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

GraphNeT's `ParquetWriter.merge_files` discovers the input parquet files from
the output path. It builds a master event list from all truth parquet files,
keeping both `event_no` and the source file name. The shuffle happens in
GraphNeT source code in `ParquetWriter._identify_events`:

```python
return res.to_pandas().sample(frac=1.0)
```

That means the event list is shuffled at event level before it is split into
merged parquet batches. The shuffled list is then divided into shards of
`events_per_batch` events.

Important detail: the GraphNeT shuffle does not pass an explicit `random_state`
to `pandas.DataFrame.sample`, so the event-level merge shuffle is not fixed by
the `seed=42` used later for train/val/test batch assignment.

### Train/Val/Test Split And Small Final Batch

After GraphNeT writes the merged batches, this repository builds splits in
`merge_parquet.py::build_splits`.

The split is done at merged-batch level, not individual-event level. The code
finds all merged truth batches, removes the last batch from the normal random
split pool, shuffles the remaining batch IDs with `seed=42`, and assigns:

```text
train = 80% of main batches
val   = 10% of main batches
test  = remaining main batches + the final batch
```

If the total number of events is not an exact multiple of `events_per_batch`
(default: 1024 in merge), GraphNeT's last shard can contain fewer events. This
repository always sends that final batch to `test`. It is not dropped.

The split metadata is written to:

```text
merged/split_manifest.json
```

## RobustScaler Percentiles

### Flavor-Specific Percentiles

The merge job computes p25/p50/p75 feature percentiles from each flavor's
training split:

```text
merged/train/features/
```

Only numeric feature columns are used. `event_no` and `global_event_no` are
excluded.

Flavor-specific CSVs are written directly under:

```text
/project/def-nahee/kbas/Graphnet-Applications/Metadata/RobustScaler/<MC>/
```

Name format:

```text
<geometry>_<flavor>[_<metadata_suffix>]_train_feature_percentiles_p25_p50_p75.csv
```

Example:

```text
102_string_electron_Emax1e6_train_feature_percentiles_p25_p50_p75.csv
```

### Mixed-Flavor Percentiles

`compute_mixed_percentiles.py` combines training features across the requested
flavors. By default these are `Muon`, `Electron`, `Tau`, and `NC`. One command
processes one geometry and produces all mixed scalers needed by the routing
pipeline:

```bash
python3 compute_mixed_percentiles.py \
  --mc 340StringMC \
  --geometry 102_string_emax1e6
```

The command first computes one category-free scaler from all training events.
This scaler is shared by all three classification methods for that geometry.
It then computes one scaler for every routed reconstruction class in:

```text
category1_isMuonCC
category2_tauCC_others_muonCC
category_3_contains_muon
```

Category-specific inputs come from the categorized `train/features/`
directories. Flavor/class combinations whose path is `does_not_exist` in
`paths.py` are skipped. Each routed class scaler is shared by its energy,
zenith, and azimuth reconstruction models.

Outputs are written under:

```text
/project/def-nahee/kbas/Graphnet-Applications/Metadata/RobustScaler/<MC>/mixed/<geometry>/
```

For each geometry, the output layout is:

```text
classification/
  train_feature_percentiles_p25_p50_p75.csv

category1_isMuonCC/
  class_0_not_muon_cc/train_feature_percentiles_p25_p50_p75.csv
  class_1_muon_cc/train_feature_percentiles_p25_p50_p75.csv

category2_tauCC_others_muonCC/
  class_0_tau_cc/train_feature_percentiles_p25_p50_p75.csv
  class_1_electron_cc_or_nc/train_feature_percentiles_p25_p50_p75.csv
  class_2_muon_cc/train_feature_percentiles_p25_p50_p75.csv

category_3_contains_muon/
  class_0_no_muon/train_feature_percentiles_p25_p50_p75.csv
  class_1_contains_muon/train_feature_percentiles_p25_p50_p75.csv
```

This is eight mixed percentile CSVs per geometry. Use these geometry keys for
the current energy-filtered String340MC datasets:

```text
full_geometry_emax1e6
160_string_emax1e6
102_string_emax1e6
```

The script requires an environment containing both Pandas and Polars, such as
the existing `.venv_try` environment used for Parquet inspection.

## Categorized Parquet Outputs

Categorized Parquet jobs read the raw `truth/` and `features/` parquet files,
filter by each category value, then build train/val/test splits for that
category. They do not write `CategoryInformation` CSVs.

For example:

```text
categorized/category1_isMuonCC/category0/train/truth/
categorized/category1_isMuonCC/category0/train/features/
categorized/category1_isMuonCC/category1/train/truth/
categorized/category1_isMuonCC/category1/train/features/
```

Each category value directory also gets a `split_manifest.json`.
