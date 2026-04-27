# =============================================================================
# Notes
# =============================================================================
# triggering is after pmt response. but still, both will be online, right? I m

# =============================================================================
# Standard library
# =============================================================================
import h5py  # must be imported before icecube to avoid HDF5 version conflict
import argparse
import glob
import os
import re
import sys
import random
import time
from os.path import expandvars
from pathlib import Path

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


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Apply PMT response to a single I3 file (one SLURM array task)")
    ap.add_argument("--flavor",   required=True, help="Particle flavor (e.g. Muon, Electron, Tau, NC)")
    ap.add_argument("--geometry", required=True, help="Geometry key (e.g. strings_102_40m)")
    ap.add_argument("--mc",       required=True, help="MC dataset name (e.g. SPRING2026MC)")
    ap.add_argument("--indir",    required=True, help="Input directory containing I3Photons files")
    ap.add_argument("--pattern",  required=True, help="Glob pattern for input files (e.g. *.i3.gz)")
    ap.add_argument("--outdir",   required=True, help="Output directory for PMT response files")
    ap.add_argument("--logdir",   required=True, help="Directory for per-file log files")
    ap.add_argument("--gcd",      required=True, help="GCD file path")
    ap.add_argument("--task-id",  type=int, default=None, help="Override task index (for local testing)")
    args = ap.parse_args()

    # --- resolve task ID ---
    if args.task_id is not None:
        task_id = args.task_id
    else:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
        if task_id < 0:
            print("ERROR: SLURM_ARRAY_TASK_ID not set. Use --task-id for local testing.")
            return 2

    # --- collect input files ---
    indir = Path(args.indir).resolve()
    inputs = sorted(indir.rglob(args.pattern))
    if not inputs:
        print(f"ERROR: No files found in {indir} matching {args.pattern}")
        return 3
    if not (0 <= task_id < len(inputs)):
        print(f"ERROR: task_id={task_id} out of range (0..{len(inputs) - 1})")
        return 4

    infile  = inputs[task_id]
    outdir  = Path(args.outdir).resolve()
    logdir  = Path(args.logdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    rel      = infile.relative_to(indir)
    parts    = list(rel.parts)
    suffixes = "".join(infile.suffixes)
    parts[-1] = parts[-1][:-len(suffixes)] if suffixes else infile.stem
    stem     = "_".join(parts)
    flavor_l = args.flavor.lower()
    outfile  = outdir / f"{flavor_l}_{stem}.i3.gz"
    logfile  = logdir / f"{flavor_l}_{stem}.out"

    # skip only if both exist
    if outfile.exists() and logfile.exists():
        print(f"[skip] both exist: {outfile.name}, {logfile.name}")
        return 0

    # --- redirect stdout/stderr to per-file log ---
    log_fh = open(logfile, "w")
    sys.stdout = log_fh
    sys.stderr = log_fh

    t_start = time.time()

    print(f"=== PMT RESPONSE JOB STARTED ===")
    print(f"array_job_id  : {os.environ.get('SLURM_ARRAY_JOB_ID', 'unknown')}")
    print(f"array_task_id : {os.environ.get('SLURM_ARRAY_TASK_ID', 'unknown')}")
    print(f"job_id        : {os.environ.get('SLURM_JOB_ID', 'unknown')}")
    print(f"task_id       : {task_id} / {len(inputs) - 1}")
    print(f"flavor        : {args.flavor}")
    print(f"geometry      : {args.geometry}")
    print(f"mc            : {args.mc}")
    print(f"infile        : {infile}")
    print(f"outfile       : {outfile}")
    print(f"logfile       : {logfile}")
    print(f"gcd           : {args.gcd}")
    log_fh.flush()

    print("[CHECKPOINT 1] Job configuration resolved, log started successfully")

    # --- tray configuration ---
    m = re.search(r"gen_(\d+)", infile.name)
    if not m:
        raise ValueError(f"Could not parse run number from filename: {infile.name}")
    runnumber = int(m.group(1))

    pulsesep = 0.2
    nDOMs    = 1

    randomService = phys_services.I3SPRNGRandomService(
        seed=1234567, nstreams=10000000, streamnum=runnumber
    )

    print("[CHECKPOINT 2] Tray configuration set, random service initialized successfully")

    # --- tray setup ---
    tray = I3Tray()
    tray.context["I3RandomService"] = randomService
    tray.AddModule("I3Reader", "reader", FilenameList=[args.gcd, str(infile)])

    print("[CHECKPOINT 3] Tray initialized, I3Reader added successfully")

    tray.AddModule(DOMSimulation, 'DOMLauncher',
                   input_map='Accepted_PulseMap',
                   output_map='PMT_Response',
                   random_service=randomService, min_time_sep=pulsesep, split_doms=True,
                   use_dark=True, dark_map='Noise_Dark', use_k40=True, k40_map='Noise_K40')

    tray.AddModule(HitCountCheck, "hitcheck", NHits=5)

    print("[CHECKPOINT 4] HitCountCheck filter added successfully")

    tray.AddModule(DOMTrigger, "DOMTrigger", trigger_map="triggerpulsemap")

    print("[CHECKPOINT 5] DOMTrigger added successfully")


    tray.AddModule(
        DetectorTrigger,
        "PONE_Trigger",
        output="_3PMT_1DOM",
        OMPMTCoinc=3,
        FullDetectorCoincidenceN=nDOMs,
        CutOnTrigger=False,
        EventLength=10000,
        TriggerTime=2000,
        PulseSeriesIn="PMT_Response",
        PulseSeriesOut="EventPulseSeries",)    
    
    tray.AddModule(
        DetectorTrigger,
        "PONE_Trigger_nonoise",
        output="_3PMT_1DOM_nonoise",
        OMPMTCoinc=3,
        FullDetectorCoincidenceN=nDOMs,
        CutOnTrigger=False,
        EventLength=10000,
        TriggerTime=2000,
        PulseSeriesIn="PMT_Response_nonoise",
        PulseSeriesOut="EventPulseSeries_nonoise",)   
    
    print("[CHECKPOINT 6] DetectorTriggers added successfully")


    tray.AddModule(
       "I3Writer",
       "writer",
       Filename=str(outfile), 
       Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Simulation, icetray.I3Frame.Physics],
       )
    
    print("[CHECKPOINT 7] Writer added successfully")

    try:
        tray.Execute()
        tray.Finish()
        elapsed = time.time() - t_start
        print(f"=== SUCCESS  elapsed={elapsed:.1f}s ===")
        print(f"outfile : {outfile}")
    except Exception as e:
        import traceback
        elapsed = time.time() - t_start
        print(f"=== FAILED  elapsed={elapsed:.1f}s ===")
        print(f"ERROR: {e}")
        traceback.print_exc()
        log_fh.flush()
        log_fh.close()
        return 1

    log_fh.flush()
    log_fh.close()
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
