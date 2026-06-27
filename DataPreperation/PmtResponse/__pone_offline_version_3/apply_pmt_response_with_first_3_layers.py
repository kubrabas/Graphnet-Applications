# =============================================================================
# Notes
# =============================================================================

# =============================================================================
# Standard library
# =============================================================================
import h5py  # must be imported before icecube to avoid HDF5 version conflict
import argparse
import importlib.util
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# =============================================================================
# IceCube / IceTray
# =============================================================================
from icecube.icetray import I3Tray, I3Units, I3Frame, OMKey
from icecube import icetray, dataclasses, dataio, simclasses
from icecube import phys_services, sim_services

# =============================================================================
# P-ONE offline modules
# =============================================================================
from DOM.PONEDOMLauncher import DOMSimulation
from DOM.OMAcceptance import OMAcceptance

from NoiseGenerators.DarkNoise import DarkNoise
from NoiseGenerators.K40Noise import K40Noise

# =============================================================================
# Additional imports
# =============================================================================
from icecube.dataclasses import ModuleKey
import numpy as np
from math import sqrt
from copy import deepcopy


# =============================================================================
# Print Info
# =============================================================================

print("[CHECKPOINT 0] Libraries imported successfully")

MODE = "with_first_3_layers"
PATHS_PY = Path(__file__).resolve().parents[3] / "Metadata" / "paths.py"
BERLIN_TZ = ZoneInfo("Europe/Berlin")


def berlin_timestamp() -> str:
    return datetime.now(BERLIN_TZ).strftime("%H:%M %d %B %Y")


def status_line(status: str, elapsed: float) -> str:
    return f"=== {status}  @ {berlin_timestamp()}: elapsed={elapsed:.1f}s ==="


def load_paths():
    spec = importlib.util.spec_from_file_location("paths", PATHS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bad_file_rule(infile: Path):
    target = str(infile.resolve())
    bad_i3_files = getattr(load_paths(), "BAD_I3_FILES", {})

    for flavors in bad_i3_files.values():
        for info in flavors.values():
            if target in set(map(str, info.get("no_daq_for_some_reason", set()))):
                return "skip_no_daq", None

            available_daq_counts = info.get("available_daq_counts", {})
            if target in available_daq_counts:
                return "limit_daq", int(available_daq_counts[target])

    return "normal", None

# =============================================================================
# Custom modules
# =============================================================================

GEO_DIR = Path(__file__).resolve().parents[3] / "Metadata" / "GeometryFiles"


def load_string_set(path):
    text = Path(path).read_text().strip()
    if not text:
        return set()

    lines = text.splitlines()
    strings = set()

    if len(lines) > 1 and lines[0].lower().startswith("string,"):
        for line in lines[1:]:
            strings.add(int(line.split(",", 1)[0].strip()))
        return strings

    for token in text.replace("\n", ",").split(","):
        token = token.strip()
        if token:
            strings.add(int(token))
    return strings


LAYOUT_STRINGS = {
    "102_string": load_string_set(GEO_DIR / "340StringMC" / "102_string.csv"),
    "160_string": load_string_set(GEO_DIR / "340StringMC" / "160_string.csv"),
    "340_string": load_string_set(GEO_DIR / "string_coordinates_340_string_mc.csv"),
}

LAYOUT_OUTPUT_LABELS = {
    "102_string": "102_String",
    "160_string": "160_String",
    "340_string": "340_String",
}

FULL_LAYOUT = "340_string"
SUB_LAYOUTS = ["160_string", "102_string"]
ACCEPTED_MAP_340 = "Accepted_PulseMap_340String"


def first_dom_trigger_time(pulse_map, strings, coincidence_n=3, coincidence_window=10.0):
    dom_hits = {}

    for omkey in pulse_map.keys():
        if omkey.string not in strings:
            continue

        dom_key = (omkey.string, omkey.om)
        pmt = int(omkey.pmt)

        if dom_key not in dom_hits:
            dom_hits[dom_key] = []

        for pulse in pulse_map[omkey]:
            dom_hits[dom_key].append((float(pulse.time), pmt))

    trigger_times = []

    for hits in dom_hits.values():
        hits.sort(key=lambda item: item[0])
        lookback = 0

        for i, (time, _) in enumerate(hits):
            while lookback < i and hits[lookback][0] < time - coincidence_window:
                lookback += 1

            pmts_in_window = {pmt for _, pmt in hits[lookback:i + 1]}
            if len(pmts_in_window) >= coincidence_n:
                trigger_times.append(int(time))
                break

    if not trigger_times:
        return None

    return min(trigger_times)


def add_trigger_flags(frame):
    if frame.Stop != icetray.I3Frame.DAQ:
        return True

    for layout, strings in LAYOUT_STRINGS.items():
        pmt_response = frame[f"PMT_Response_{LAYOUT_OUTPUT_LABELS[layout]}"]
        trigger_time = first_dom_trigger_time(pmt_response, strings)
        triggered = trigger_time is not None

        frame[f"triggered_{layout}"] = dataclasses.I3Double(float(triggered))
        frame[f"trigger_time_{layout}"] = dataclasses.I3Double(
            float(trigger_time) if triggered else -1.0
        )

    return True


def drop_stale_keys(frame):
    keys = [
        "Accepted_PulseMap",
        ACCEPTED_MAP_340,
        "Noise_Dark",
        "Noise_K40",
        "PMT_Response",
        "PMT_Response_nonoise",
    ]
    for label in LAYOUT_OUTPUT_LABELS.values():
        keys.extend([
            f"Noise_Dark_{label}",
            f"Noise_K40_{label}",
            f"PMT_Response_{label}",
            f"PMT_Response_{label}_nonoise",
            f"PMT_Response_nonoise_{label}",
        ])

    for key in keys:
        if key in frame:
            frame.Delete(key)


def rename_nonoise_response_maps(frame):
    if frame.Stop != icetray.I3Frame.DAQ:
        return True

    for label in LAYOUT_OUTPUT_LABELS.values():
        old_key = f"PMT_Response_{label}_nonoise"
        new_key = f"PMT_Response_nonoise_{label}"
        if old_key in frame:
            frame[new_key] = frame[old_key]
            frame.Delete(old_key)

    return True


def subset_map_by_strings(source_map, strings):
    output_map = source_map.__class__()

    for omkey in source_map.keys():
        if omkey.string in strings:
            output_map[omkey] = source_map[omkey]

    return output_map


def add_subgeometry_maps(frame):
    if frame.Stop != icetray.I3Frame.DAQ:
        return True

    source_keys = [
        "Noise_Dark",
        "Noise_K40",
        "PMT_Response",
        "PMT_Response_nonoise",
    ]

    for layout in SUB_LAYOUTS:
        strings = LAYOUT_STRINGS[layout]
        label = LAYOUT_OUTPUT_LABELS[layout]

        for source_key in source_keys:
            source_full_key = f"{source_key}_{LAYOUT_OUTPUT_LABELS[FULL_LAYOUT]}"
            target_key = f"{source_key}_{label}"
            frame[target_key] = subset_map_by_strings(frame[source_full_key], strings)

    return True


class DAQFrameLimiter(icetray.I3ConditionalModule):
    def __init__(self, ctx):
        super(DAQFrameLimiter, self).__init__(ctx)
        self.AddOutBox("OutBox")
        self.AddParameter("MaxDAQFrames", "Maximum DAQ frames to process; <=0 means unlimited", 0)

    def Configure(self):
        self.max_daq_frames = int(self.GetParameter("MaxDAQFrames"))
        self.daq_count = 0

    def Process(self):
        frame = self.PopFrame()

        if frame.Stop == icetray.I3Frame.DAQ:
            self.daq_count += 1
            if self.max_daq_frames > 0 and self.daq_count > self.max_daq_frames:
                self.RequestSuspension()
                return

        self.PushFrame(frame)

        if frame.Stop == icetray.I3Frame.DAQ and self.max_daq_frames > 0 and self.daq_count >= self.max_daq_frames:
            self.RequestSuspension()


# =============================================================================
# Per-file worker  (runs in a subprocess via ProcessPoolExecutor)
# =============================================================================

def _process_one(infile: Path, outfile: Path, logfile: Path, cfg: dict) -> tuple:
    """Process a single I3 file.

    Returns:
        ("success",  filename)
        ("failed",   filename, error_message)
    """

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_fh = None
    t_start = time.time()
    frame_counts = {"daq": 0, "simulation": 0}

    try:
        log_fh = open(logfile, "w")
        sys.stdout = log_fh
        sys.stderr = log_fh

        print("=== PMT RESPONSE JOB STARTED (with_first_3_layers) ===")
        print(f"started  : {datetime.now(BERLIN_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"infile   : {infile}")
        print(f"outfile  : {outfile}")
        print(f"logfile  : {logfile}")
        print(f"flavor   : {cfg['flavor']}")
        print(f"geometry : {cfg['geometry']}")
        print(f"mc       : {cfg['mc']}")
        print(f"gcd      : {cfg['gcd']}")
        log_fh.flush()

        print("[CHECKPOINT 1] Log started successfully")

        rule, available_daq_count = bad_file_rule(infile)
        if rule == "skip_no_daq":
            elapsed = time.time() - t_start
            print("--- Skipping: input file is listed in paths.BAD_I3_FILES as no_daq_for_some_reason")
            print(status_line("SKIPPED", elapsed))
            log_fh.flush()
            return ("skipped", infile.name, elapsed, "no_daq_for_some_reason")

        max_daq_frames = available_daq_count if rule == "limit_daq" else 0
        if max_daq_frames > 0:
            print(f"available_daq_count : {max_daq_frames}")
        else:
            print("available_daq_count : unlimited")

        m = re.search(r"gen_(\d+)", infile.name) or re.search(r"cls_(\d+)", infile.name)
        if not m:
            raise ValueError(f"Could not parse run number from filename: {infile.name}")
        runnumber = int(m.group(1))

        pulsesep = 0.2
        randomService = phys_services.I3SPRNGRandomService(
            seed=1234567, nstreams=10000000, streamnum=runnumber
        )

        print("[CHECKPOINT 2] Tray configuration set, random service initialized successfully")

        tray = I3Tray()
        tray.context["I3RandomService"] = randomService
        tray.AddModule("I3Reader", "reader", FilenameList=[cfg['gcd'], str(infile)])
        tray.AddModule(DAQFrameLimiter, "DAQFrameLimiter", MaxDAQFrames=max_daq_frames)

        tray.AddModule(drop_stale_keys, "DropStaleKeys", Streams=[icetray.I3Frame.DAQ])

        print("[CHECKPOINT 3] Tray initialized, I3Reader and DropStaleKeys added successfully")

        full_label = LAYOUT_OUTPUT_LABELS[FULL_LAYOUT]
        dark_map_340 = f"Noise_Dark_{full_label}"
        k40_map_340 = f"Noise_K40_{full_label}"
        response_map_340 = f"PMT_Response_{full_label}"

        tray.AddModule(OMAcceptance, f"OMAcceptance_{full_label}",
                       input_map="I3Photons", output_map=ACCEPTED_MAP_340,
                       random_service=randomService,
                       drop_empty=False)

        tray.AddModule(DarkNoise, f"AddDarkNoise_{full_label}",
                       input_map=ACCEPTED_MAP_340, output_map=dark_map_340,
                       random_service=randomService, gcd_file=cfg['gcd'])

        tray.AddModule(K40Noise, f"AddK40Noise_{full_label}",
                       input_map=ACCEPTED_MAP_340, output_map=k40_map_340,
                       random_service=randomService, gcd_file=cfg['gcd'])

        tray.AddModule(DOMSimulation, f"DOMLauncher_{full_label}",
                       input_map=ACCEPTED_MAP_340,
                       output_map=response_map_340,
                       random_service=randomService, min_time_sep=pulsesep, split_doms=True,
                       use_dark=True, dark_map=dark_map_340, use_k40=True, k40_map=k40_map_340)

        print("[CHECKPOINT 4] OMAcceptance, DarkNoise, K40Noise, DOMSimulation added for full 340-string layout")

        tray.AddModule(rename_nonoise_response_maps, "RenameNonoiseResponseMaps", Streams=[icetray.I3Frame.DAQ])
        tray.AddModule(add_subgeometry_maps, "AddSubgeometryMaps", Streams=[icetray.I3Frame.DAQ])
        tray.AddModule(add_trigger_flags, "AddTriggerFlags", Streams=[icetray.I3Frame.DAQ])

        print("[CHECKPOINT 5] Subgeometry maps, response map names, and trigger flags added successfully")

        def count_output_frames(frame):
            if frame.Stop == icetray.I3Frame.DAQ:
                frame_counts["daq"] += 1
            elif frame.Stop == icetray.I3Frame.Simulation:
                frame_counts["simulation"] += 1
            return True

        tray.AddModule(
            count_output_frames, "OutputFrameCounter",
            Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Simulation],
        )

        tray.AddModule(
            "I3Writer", "writer",
            Filename=str(outfile),
            Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Simulation],
        )

        print("[CHECKPOINT 6] Writer added successfully")

        tray.Execute()
        tray.Finish()

        elapsed = time.time() - t_start
        print(status_line("SUCCESS", elapsed))
        print(f"frames_to_writer : DAQ={frame_counts['daq']}  Simulation={frame_counts['simulation']}")
        print(f"outfile : {outfile}")
        log_fh.flush()

        return ("success", infile.name, time.time() - t_start, frame_counts["daq"], frame_counts["simulation"])

    except Exception as e:
        import traceback as _tb
        elapsed = time.time() - t_start
        if log_fh and not log_fh.closed:
            try:
                print(status_line("FAILED", elapsed))
                print(f"ERROR: {e}")
                print(f"frames_to_writer_before_failure : DAQ={frame_counts['daq']}  Simulation={frame_counts['simulation']}")
                _tb.print_exc()
                log_fh.flush()
            except Exception:
                pass

        try:
            if log_fh and not log_fh.closed:
                log_fh.close()
        except Exception:
            pass
        log_fh = None  # prevent double-close in finally

        # Keep the failed log for diagnosis, but remove the failed output.
        for p in (outfile,):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

        return ("failed", infile.name, str(e), elapsed, frame_counts["daq"], frame_counts["simulation"])

    finally:
        if log_fh is not None and not log_fh.closed:
            try:
                log_fh.close()
            except Exception:
                pass
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Apply PMT response (with first 3 layers) — parallel, single SLURM node"
    )
    ap.add_argument("--flavor",   required=True, help="Particle flavor (e.g. Muon, Electron, Tau, NC)")
    ap.add_argument("--geometry", required=True, help="Geometry key (e.g. strings_102_40m)")
    ap.add_argument("--mc",       required=True, help="MC dataset name (e.g. SPRING2026MC)")
    ap.add_argument("--indir",    required=True, help="Input directory containing I3Photons files")
    ap.add_argument("--pattern",  required=True, help="Glob pattern for input files (e.g. *.i3.gz)")
    ap.add_argument("--outdir",   required=True, help="Output directory for PMT response files")
    ap.add_argument("--logdir",   required=True, help="Directory for per-file log files")
    ap.add_argument("--gcd",      required=True, help="GCD file path")
    ap.add_argument("--nworkers", type=int, default=None,
                    help="Parallel workers (default: $SLURM_CPUS_PER_TASK or 4)")
    args = ap.parse_args()

    indir  = Path(args.indir).resolve()
    outdir = Path(args.outdir).resolve()
    logdir = Path(args.logdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(indir.rglob(args.pattern))
    if not inputs:
        print(f"ERROR: No files found in {indir} matching {args.pattern}")
        return 3

    flavor_l = args.flavor.lower()

    # Build task list with output paths for every input file
    tasks = []
    for infile in inputs:
        rel      = infile.relative_to(indir)
        parts    = list(rel.parts)
        suffixes = "".join(infile.suffixes)
        parts[-1] = parts[-1][:-len(suffixes)] if suffixes else infile.stem
        stem     = "_".join(parts)
        outfile  = outdir / f"{flavor_l}_{stem}.i3.gz"
        logfile  = logdir / f"{flavor_l}_{stem}.out"
        tasks.append((infile, outfile, logfile))

    n_cleaned_log = n_cleaned_out = 0
    to_process = []
    for inf, outf, logf in tasks:
        has_out = outf.exists()
        has_log = logf.exists()
        if has_out and has_log:
            with open(logf, "r", errors="ignore") as fh:
                if any(("SKIPPED" in line or "SUCCESS" in line) for line in fh):
                    continue
            logf.unlink()
            outf.unlink()
        if has_log and not has_out:
            logf.unlink()
            n_cleaned_log += 1
        elif has_out and not has_log:
            outf.unlink()
            n_cleaned_out += 1
        to_process.append((inf, outf, logf))

    n_already_done = len(tasks) - len(to_process)

    print(f"Total files        : {len(tasks)}")
    print(f"Already done (skip): {n_already_done}")
    print(f"Cleaned (log only) : {n_cleaned_log}")
    print(f"Cleaned (out only) : {n_cleaned_out}")
    print(f"To process         : {len(to_process)}")

    if not to_process:
        print("Nothing to do.")
        return 0

    nworkers = args.nworkers or int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    print(f"Workers            : {nworkers}")
    sys.stdout.flush()

    job_id           = os.environ.get("SLURM_JOB_ID", "local")
    job_start_dt     = datetime.now(BERLIN_TZ)
    job_date         = job_start_dt.strftime("%Y-%m-%d")
    general_logdir   = logdir
    general_logdir.mkdir(parents=True, exist_ok=True)
    general_log_path = general_logdir / f"summary_{job_date}_job_{job_id}.log"
    t_job_start      = time.time()

    cfg = vars(args)
    n_success = n_failed = n_skipped = 0

    with open(general_log_path, "w") as glog:
        def _glog(msg=""):
            glog.write(msg + "\n")
            glog.flush()

        _glog("=== PMT RESPONSE JOB ===")
        _glog(f"job_id   : {job_id}")
        _glog(f"started  : {job_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        _glog(f"flavor   : {args.flavor}")
        _glog(f"geometry : {args.geometry}")
        _glog(f"mc       : {args.mc}")
        _glog(f"mode     : {MODE}")
        _glog(f"outdir   : {outdir}")
        _glog(f"logdir   : {logdir}")
        _glog(f"total    : {len(tasks)}")
        _glog(f"skipped  : {n_already_done}")
        _glog(f"cleaned  : log_only={n_cleaned_log}  out_only={n_cleaned_out}")
        _glog(f"to_proc  : {len(to_process)}")
        _glog(f"workers  : {nworkers}")
        _glog()

        with ProcessPoolExecutor(max_workers=nworkers) as executor:
            futures = {
                executor.submit(_process_one, inf, outf, logf, cfg): inf.name
                for inf, outf, logf in to_process
            }
            for future in as_completed(futures):
                result  = future.result()
                status  = result[0]
                fname   = result[1]
                ts      = datetime.now(BERLIN_TZ).strftime('%H:%M:%S')
                if status == "success":
                    n_success += 1
                    elapsed = result[2] if len(result) > 2 else 0.0
                    daq = result[3] if len(result) > 3 else 0
                    sim = result[4] if len(result) > 4 else 0
                    print(f"[ok]     {fname}")
                    _glog(f"[{ts}] [ok]     {fname}  ({elapsed:.1f}s)  frames_to_writer: DAQ={daq} Simulation={sim}")
                elif status == "skipped":
                    n_skipped += 1
                    elapsed = result[2] if len(result) > 2 else 0.0
                    reason = result[3] if len(result) > 3 else "skipped"
                    print(f"[skipped] {fname}  ({reason})")
                    _glog(f"[{ts}] [skipped] {fname}  ({elapsed:.1f}s)  {reason}")
                else:
                    n_failed += 1
                    err     = result[2] if len(result) > 2 else "unknown"
                    elapsed = result[3] if len(result) > 3 else 0.0
                    daq = result[4] if len(result) > 4 else 0
                    sim = result[5] if len(result) > 5 else 0
                    print(f"[failed] {fname}  ({err})")
                    _glog(f"[{ts}] [failed] {fname}  ({elapsed:.1f}s)  frames_to_writer_before_failure: DAQ={daq} Simulation={sim} -- {err}")
                sys.stdout.flush()

        total_elapsed = time.time() - t_job_start
        print(f"\nDone.  success={n_success}  skipped={n_skipped}  failed={n_failed}")
        _glog()
        _glog("=== FINAL SUMMARY ===")
        _glog(f"finished : {datetime.now(BERLIN_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        _glog(f"success  : {n_success}")
        _glog(f"skipped  : {n_skipped}")
        _glog(f"failed   : {n_failed}")
        _glog(f"elapsed  : {total_elapsed:.1f}s")
        _glog(f"log      : {general_log_path}")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
