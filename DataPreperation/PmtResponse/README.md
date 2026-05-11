# PMT Response Workflow

This folder contains the scripts used to submit PMT response production jobs, run the per file processing, collect logs, and inspect the run afterward. This note focuses on the script flow and the logging behavior, not on the physics of the PMT response step.

## Main Files

`submit_pmt_response.py`

This is the submission entry point. It reads dataset paths from `Metadata/paths.py`, resolves the selected MC sample, geometry, flavor list, input format, GCD file, output directory, and log directory. It does not process events directly. Its job is to prepare the job configuration and submit one SLURM job per flavor.

`apply_pmt_response_with_first_3_layers.py`

Worker script used when the first 3 layers are included. It processes all input files for one flavor inside a single SLURM job, using multiple local worker processes.

`apply_pmt_response_without_first_3_layers.py`

Worker script used when the first 3 layers are not included. The parallel execution, output naming, logging, and error handling are the same as in the other worker script.

`QA.ipynb`

Notebook for post run checks based on the log files. It builds one dataframe per flavor and reports the final status, DAQ frames seen by the writer counter, output existence, output size, and error information for each file.

## Submission Flow

1. The submit script is started with an MC name, geometry, flavor selection, and first 3 layers mode.
2. It loads `Metadata/paths.py`.
3. It selects the input table for the MC sample.
4. It resolves the geometry folder name and the GCD file.
5. It expands `all` into `Muon`, `Electron`, `Tau`, and `NC`.
6. For each flavor, it checks whether the input path exists and whether matching input files are present.
7. If no files are found, that flavor is skipped.
8. If files are found, one SLURM job is submitted for that flavor.
9. The SLURM wrapper selects the correct worker script.
10. The worker script processes all files for that flavor in parallel.

## Worker Selection

The submit script passes a first 3 layers mode value to the SLURM wrapper.

Value `1` selects:

```text
apply_pmt_response_with_first_3_layers.py
```

Value `0` selects:

```text
apply_pmt_response_without_first_3_layers.py
```

The worker script itself only receives paths and runtime settings. It does not decide which mode should be used.

## Output Layout

Output files are written under:

```text
/scratch/kbas/{MCFolder}/{GeometryFolder}/{Flavor}_PMT_Response
```

Per file logs are written under:

```text
/scratch/kbas/{MCFolder}/Logs/{Flavor}_pmt_response_{GeometryFolder}
```

The job level summary log is written under:

```text
/scratch/kbas/{MCFolder}/Logs/{Flavor}_pmt_response_{GeometryFolder}/summary_{YYYY-MM-DD}_job_{job_id}.log
```

The date in the summary log filename is the job start date in Berlin time.

Some scripts may record paths under `/home/kbas/scratch`, while the same storage is often accessed as `/scratch/kbas` on the login node. The QA notebook checks the configured output directory as a fallback when the logged output path is not found.

## Skip and Cleanup Logic

At the start of each worker job, before any processing begins, the worker
compares the expected output files against the expected log files for every
input file in the task list.

For each input file the worker computes the expected output path and the
expected per-file log path. It then checks which of these already exist on
disk and applies the following rules:

- Both output and log exist → file is considered done, skip it.
- Log exists but output is missing → the log is a leftover from a previous
  failed or interrupted run. The log is deleted and the file is added to the
  work queue.
- Output exists but log is missing → the output is a leftover without a
  matching record. The output is deleted and the file is added to the work
  queue.
- Neither exists → the file is added to the work queue normally.

The terminal output and the summary log both report the counts before any
processing starts:

```text
Total files        : 9977
Already done (skip): 6000
Cleaned (log only) : 3
Cleaned (out only) : 1
To process         : 3977
```

`To process` includes the cleaned files.

## Logging Model

There are two useful log levels.

The per file log has this form:

```text
{flavor_lower}_{input_stem}.out
```

Each input file gets its own log file. During processing, both standard output and standard error are redirected into that file. Checkpoints, IceTray output, Python errors, and tracebacks are all expected to appear there.

The job summary log has this form:

```text
summary_{YYYY-MM-DD}_job_{job_id}.log
```

This log is written by the parent process. It starts with a header that includes the skip and cleanup counts, then records one line per completed file with the final status, runtime, and DAQ frame count from the writer counter.

The SLURM wrapper standard output and standard error are disabled, so wrapper level echo output is not kept in a separate SLURM log file.

## Per File Log Content

At the start of each per file log, the worker writes the basic run information:

```text
infile
outfile
logfile
flavor
geometry
mc
gcd
```

During tray setup, checkpoint lines are written. These are useful for seeing how far the file got before a failure.

On success, the log ends with:

```text
SUCCESS
elapsed
frames_to_writer
outfile
```

On failure, the log ends with:

```text
FAILED
elapsed
ERROR
frames_to_writer_before_failure
Traceback
```

The `DAQ` value in `frames_to_writer` is the number of DAQ frames that reached the counter placed immediately before the writer module. For failed files, `frames_to_writer_before_failure` gives the same information at the point where the exception was caught.

This makes it possible to distinguish files that failed before writing any useful DAQ frames from files that failed after at least some events reached the writer stage.

## Error Handling

If a normal Python or IceTray exception happens while processing an input file, the worker catches it.

When that happens:

1. The per file log is kept.
2. The error message is written to the log.
3. The traceback is written to the log.
4. The DAQ and Simulation frame counts seen before failure are written to the log.
5. Any partially produced output file is removed.
6. Other files in the same flavor job continue processing.

At the end of the job, the worker exits successfully only if all files succeeded. If one or more files failed, the worker returns a failure code. Successful output files from the same job are still kept.

Hard crashes, node kills, memory kills, or failures that do not become Python exceptions may stop the process before the failure block runs. In those cases, the final failure lines may be missing from the per file log.

## Known Bad I3 Files

The PMT workers check `Metadata/paths.py` for `BAD_I3_FILES` before running each input file.

- Files listed under `no_daq_for_some_reason` are skipped and reported as `skipped`, not `failed`.
- Files listed under `available_daq_counts` are processed only up to the recorded number of safe DAQ frames.
- Files not listed there use the normal unlimited processing path.

Skipped files keep a per file log ending with `SKIPPED`, and the job summary log includes a separate `skipped` count.

## Frame Filtering and Empty Output Files

Two modules in the pipeline can drop DAQ frames before they reach the writer.

`HitCountCheck` drops any frame whose `PMT_Response` contains fewer than 5 unique OMs. The frame is discarded without being pushed downstream.

`DetectorTrigger` with `CutOnTrigger=True` drops frames that do not satisfy the detector-level trigger condition.

If all frames in a file are dropped by either of these filters, the tray still completes without error. The output `.i3.gz` file is written and the job reports `success`, but the file contains only GCD frames and zero DAQ events. This appears in the per-file log as:

```text
frames_to_writer : DAQ=0  Simulation=0
```

and in the job summary log as a `success` line with `DAQ=0`. These files are not failures and do not produce a traceback. They can be identified in the QA notebook by filtering for `status == "success"` and `daq_frames_to_writer == 0`.

## QA Notebook

`QA.ipynb` reads the per file logs and builds dataframes for the selected MC and geometry.

The first cell contains imports. The second cell contains the run configuration:

```text
MC_NAME
GEOMETRY
SCRATCH_BASE
FLAVORS
```

The notebook creates:

```text
Muon_df
Electron_df
Tau_df
NC_df
all_df
```

Important columns include:

```text
status
daq_frames_to_writer
simulation_frames_to_writer
elapsed_s
error
has_traceback
output_exists
output_size_mb
log_size_mb
log_path
output_path
```

`status` can be:

```text
success
failed
unknown
```

`unknown` means the log did not contain a final success or failed line. This can happen if the job is still running, the log is incomplete, or the process ended unexpectedly.

## Failed DAQ Summary

The final QA cell summarizes failed files by flavor. It answers the question:

```text
Among failed files, how many had zero DAQ frames and how many reached at least one DAQ frame before failing?
```

The summary columns are:

```text
failed_total
failed_zero_daq
failed_positive_daq
failed_missing_daq
failed_positive_daq_fraction
failed_zero_daq_fraction
failed_daq_min
failed_daq_median
failed_daq_max
```

The notebook also displays `failed_zero_daq_files`, which lists the failed files with zero or missing DAQ frame counts.

## Usual Check Order

After a run, start with the job summary log to see which files succeeded or failed and how many DAQ frames reached the writer counter.

For failed files, open the matching per file `.out` log. That file contains the traceback and the detailed checkpoint history.

For a full run overview, use `QA.ipynb`. It is the quickest way to compare flavors, count failures, find zero DAQ failures, and check output file sizes.
