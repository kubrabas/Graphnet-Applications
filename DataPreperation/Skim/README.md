# Skim

Utilities for reducing P-ONE I3/GCD files to a selected subset of detector strings.

## Files

- `FilterFrame.py`: IceTray module that applies the actual filtering.
- `trim_GCD.py`: creates a reduced GCD file for a selected string layout.
- `trim_I3.py`: skims one I3 photon file, intended for Slurm array jobs.
- `submit_skim_I3.py`: submits `trim_I3.py` array jobs to Slurm.
- `manual_skim.ipynb`: notebook for local/manual checks.

## String Selection

String selections are read from CSV/text files containing string IDs. The scripts extract all integers from the file, keep them in first-seen order, and pass them to `FilterFrame` as `AllowedStrings`.

Example:

```bash
python3 trim_GCD.py \
  --selection /project/def-nahee/kbas/Graphnet-Applications/Metadata/GeometryFiles/Spring2026MC/strings_102_40m.csv
```

## Optional OM Exclusion

You can also drop specific OM numbers within the selected strings. This is optional; if omitted, the old string-only behavior is preserved.

```bash
python3 trim_GCD.py \
  --selection /path/to/strings.csv \
  --exclude-oms 1,2,3
```

For Slurm I3 skims:

```bash
python3 submit_skim_I3.py \
  --csv /path/to/strings.csv \
  --flavor Muon \
  --exclude-oms 1,2,3
```

`submit_skim_I3.py` exports the OM list to the worker as `EXCLUDE_OMS`, and `trim_I3.py` passes it to `FilterFrame` as `ExcludedOMs`.
The submit script normalizes comma/space-separated input to a Slurm-safe format internally, so `--exclude-oms 1,2,3` is safe to use.

## Bad I3 Files

`trim_I3.py` checks `Metadata/paths.py` for known problematic input files:

- files listed under `no_daq_for_some_reason` are skipped without writing an output I3 file.
- files listed under `available_daq_counts` are processed only up to the recorded number of safe DAQ frames.
- files not listed there use the normal unlimited processing path.


Tested On:
Not Tested On: