# =============================================================================
# Notes
# =============================================================================
# triggering is after pmt response. but still, both will be online, right? I m

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

from Trigger.DOMTrigger import DOMTrigger
from Trigger.DetectorTrigger import DetectorTrigger

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
PATHS_PY = Path(__file__).resolve().parents[2] / "Metadata" / "paths.py"
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

class HitCountCheck(icetray.I3Module):
    def __init__(self, context):
        super(HitCountCheck, self).__init__(context)
        self.AddParameter("NHits", "Minimum number of unique OMs required to pass frame", 5)

    def Configure(self):
        self.NHits = self.GetParameter("NHits")

    def DAQ(self, frame):
        unique_oms = set((k.string, k.om) for k in frame['PMT_Response'].keys())
        if len(unique_oms) < self.NHits:
            return False
        self.PushFrame(frame)


class FixTriggerMap(icetray.I3Module):
    """
    DOMSimulation writes triggerpulsemap with per-PMT keys and wrong pulse.width.
    This module rebuilds it as per-DOM keys with pulse.width = PMT index,
    which is what DOMTrigger expects.
    """
    def __init__(self, context):
        super().__init__(context)
        self.AddOutBox("OutBox")

    def Configure(self):
        pass

    def DAQ(self, frame):
        bad_map = frame["triggerpulsemap"]
        dom_map = dataclasses.I3RecoPulseSeriesMap()

        for omkey in bad_map.keys():
            pmt     = omkey.pmt
            dom_key = OMKey(omkey.string, omkey.om, 0)
            if dom_key not in dom_map.keys():
                dom_map[dom_key] = dataclasses.I3RecoPulseSeries()
            for pulse in bad_map[omkey]:
                new_pulse        = dataclasses.I3RecoPulse()
                new_pulse.time   = pulse.time
                new_pulse.charge = pulse.charge
                new_pulse.width  = float(pmt)
                dom_map[dom_key].append(new_pulse)

        for dom_key in dom_map.keys():
            pulses = sorted(dom_map[dom_key], key=lambda p: p.time)
            series = dataclasses.I3RecoPulseSeries()
            for p in pulses:
                series.append(p)
            dom_map[dom_key] = series

        frame.Delete("triggerpulsemap")
        frame["triggerpulsemap"] = dom_map
        self.PushFrame(frame)


def drop_stale_keys(frame):
    for key in ["Accepted_PulseMap", "Noise_Dark", "Noise_K40"]:
        if key in frame:
            frame.Delete(key)


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
        nDOMs    = 1

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

        tray.AddModule(OMAcceptance, 'OMAcceptance',
                       input_map="I3Photons", output_map='Accepted_PulseMap',
                       random_service=randomService,
                       drop_empty=True)

        tray.AddModule(DarkNoise, 'AddDarkNoise',
                       input_map='Accepted_PulseMap', output_map='Noise_Dark',
                       random_service=randomService, gcd_file=cfg['gcd'])

        tray.AddModule(K40Noise, 'AddK40Noise',
                       input_map='Accepted_PulseMap', output_map='Noise_K40',
                       random_service=randomService, gcd_file=cfg['gcd'])

        print("[CHECKPOINT 4] OMAcceptance, DarkNoise, K40Noise added successfully")

        tray.AddModule(DOMSimulation, 'DOMLauncher',
                       input_map='Accepted_PulseMap',
                       output_map='PMT_Response',
                       random_service=randomService, min_time_sep=pulsesep, split_doms=True,
                       use_dark=True, dark_map='Noise_Dark', use_k40=True, k40_map='Noise_K40')

        tray.AddModule(HitCountCheck, "hitcheck", NHits=5)

        print("[CHECKPOINT 5] DOMSimulation and HitCountCheck added successfully")

        tray.AddModule(FixTriggerMap, "FixTriggerMap")
        tray.AddModule(DOMTrigger, "DOMTrigger", trigger_map="triggerpulsemap")

        print("[CHECKPOINT 6] FixTriggerMap and DOMTrigger added successfully")

        tray.AddModule(
            DetectorTrigger, "PONE_Trigger",
            output="_3PMT_1DOM",
            OMPMTCoinc=3,
            FullDetectorCoincidenceN=nDOMs,
            CutOnTrigger=True,
            EventLength=10000,
            TriggerTime=2000,
            PulseSeriesIn="PMT_Response",
            PulseSeriesOut="EventPulseSeries",
        )

        tray.AddModule(
            DetectorTrigger, "PONE_Trigger_nonoise",
            output="_3PMT_1DOM_nonoise",
            OMPMTCoinc=3,
            FullDetectorCoincidenceN=nDOMs,
            CutOnTrigger=False,
            EventLength=10000,
            TriggerTime=2000,
            PulseSeriesIn="PMT_Response_nonoise",
            PulseSeriesOut="EventPulseSeries_nonoise",
        )

        print("[CHECKPOINT 7] DetectorTriggers added successfully")

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

        print("[CHECKPOINT 8] Writer added successfully")

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
            continue
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
