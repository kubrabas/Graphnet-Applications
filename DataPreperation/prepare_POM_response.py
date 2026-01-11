#!/usr/bin/env python

import argparse
import os, re, glob

from icecube.icetray import I3Tray
from icecube import phys_services, icetray, dataclasses

from DOM.DOMAcceptance import DOMAcceptance
from DOM.PONEDOMLauncher import DOMSimulation

from NoiseGenerators.DarkNoise import DarkNoise
from NoiseGenerators.K40Noise import K40Noise

from PulseCleaning.CausalHits import CausalPulseCleaning
from Trigger.DOMTrigger import DOMTrigger
from Trigger.DetectorTrigger import DetectorTrigger


OUT_DIR_DEFAULT = "/project/def-nahee/kbas/POM_Response_new"
GCD_DEFAULT     = "/project/6008051/pone_simulation/GCD_Library/PONE_800mGrid.i3.gz"
PHOTON_DIR_DEF  = "/project/6008051/pone_simulation/MC10-000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim/Photon"


def guess_batch_id(infile: str) -> str:
    """
    Try to extract a "batch id" from the input filename/path.
    Priority:
      1) batch<digits> (case-insensitive)
      2) first long digit token (>=5)
      3) basename without .i3/.gz
    """
    m = re.search(r"(batch\d+)", infile, re.IGNORECASE)
    if m:
        return m.group(1)

    m = re.search(r"(\d{5,})", os.path.basename(infile))
    if m:
        return m.group(1)

    base = os.path.basename(infile)
    base = re.sub(r"\.i3(\.gz)?$", "", base)
    base = re.sub(r"\.gz$", "", base)
    return base


class DAQToPhysics(icetray.I3Module):
    """
    - Drop any incoming Physics frames (DetectorTrigger pushes empty P frames).
    - For each DAQ frame, create a Physics frame containing the keys we care about.
    - Then the writer can write only Physics frames.
    """
    def __init__(self, context):
        super(DAQToPhysics, self).__init__(context)
        self.AddParameter("keys_to_copy", "Frame object names to copy DAQ -> Physics", [])
        self.AddOutBox("OutBox")

    def Configure(self):
        self.keys_to_copy = self.GetParameter("keys_to_copy")

    def Physics(self, frame):
        # Drop DetectorTrigger's empty P frames
        return

    def DAQ(self, frame):
        p = icetray.I3Frame(icetray.I3Frame.Physics)

        # Copy event header if present (handy downstream)
        if "I3EventHeader" in frame:
            p["I3EventHeader"] = frame["I3EventHeader"]

        for k in self.keys_to_copy:
            if k in frame:
                p[k] = frame[k]

        self.PushFrame(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", default=PHOTON_DIR_DEF,
                        help="Input CLSim file OR directory containing CLSim i3(.gz) files")
    parser.add_argument("-g", "--gcdfile", default=GCD_DEFAULT,
                        help="GCD file")
    parser.add_argument("-o", "--outdir", default=OUT_DIR_DEFAULT,
                        help="Output directory")
    parser.add_argument("-r", "--runnumber", type=int, default=1,
                        help="streamnum for I3SPRNGRandomService")
    parser.add_argument("-t", "--pulsesep", type=float, default=0.2,
                        help="min_time_sep (ns) for pulse merging")
    parser.add_argument("-n", "--nDOMs", type=int, default=1,
                        help="FullDetectorCoincidenceN (keep default if you want unchanged)")
    parser.add_argument("--photon_key", default="I3Photons",
                        help="Photon series key in the CLSim file")
    args = parser.parse_args()

    # Resolve input file: if directory, pick first *.i3.gz (or *.i3) sorted
    infile = args.infile
    if os.path.isdir(infile):
        cand = sorted(glob.glob(os.path.join(infile, "*.i3.gz")) + glob.glob(os.path.join(infile, "*.i3")))
        if len(cand) == 0:
            raise RuntimeError(f"No .i3/.i3.gz files found in directory: {infile}")
        infile = cand[0]

    batch_id = guess_batch_id(infile)
    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(args.outdir, f"{batch_id}.i3.gz")

    tray = I3Tray()

    randomService = phys_services.I3SPRNGRandomService(
        seed=1234567, nstreams=10000, streamnum=args.runnumber
    )
    tray.context["I3RandomService"] = randomService

    tray.AddModule("I3Reader", "reader", FilenameList=[args.gcdfile, infile])

    # 1) Acceptance + PMT split
    tray.AddModule(
        DOMAcceptance,
        "DOMAcceptance",
        input_map=args.photon_key,
        output_map="Accepted_PulseMap",
        random_service=randomService,
        drop_empty=True,
    )

    # 2) Noise (dark + K40)
    tray.AddModule(
        DarkNoise,
        "AddDarkNoise",
        input_map="Accepted_PulseMap",
        output_map="Noise_Dark",
        random_service=randomService,
        gcd_file=args.gcdfile,
    )

    tray.AddModule(
        K40Noise,
        "AddK40Noise",
        input_map="Accepted_PulseMap",
        output_map="Noise_K40",
        random_service=randomService,
        gcd_file=args.gcdfile,
    )

    # 3) PMT response (signal + noise) + also writes PMT_Response_nonoise + triggerpulsemap
    tray.AddModule(
        DOMSimulation,
        "DOMSimulation",
        input_map="Accepted_PulseMap",
        output_map="PMT_Response",
        random_service=randomService,
        min_time_sep=args.pulsesep,   # unchanged if you keep default
        split_doms=True,
        use_dark=True,
        dark_map="Noise_Dark",
        use_k40=True,
        k40_map="Noise_K40",
    )

    # 4) Trigger -> EventPulseSeries
    tray.AddModule(
        DOMTrigger,
        "DOMTrigger",
        trigger_map="triggerpulsemap",
    )

    tray.AddModule(
        DetectorTrigger,
        "PONE_Trigger",
        output="_3PMT_2DOM",
        OMPMTCoinc=3,
        FullDetectorCoincidenceN=args.nDOMs,  # unchanged if you keep default
        CutOnTrigger=False,
        EventLength=10000,
        TriggerTime=2000,
        PulseSeriesIn="PMT_Response",
        PulseSeriesOut="EventPulseSeries",
    )

    # 5) Convert DAQ -> Physics (and drop DetectorTrigger's empty Physics frames)
    keys_to_copy = [
        "PMT_Response",
        "PMT_Response_nonoise",
        "EventPulseSeries",
        "Noise_Dark",
        "Noise_K40",
        "DetectorTriggers_3PMT_2DOM",
        "StringTriggers_3PMT_2DOM",
        "singleOMTrigger_3PMT_2DOM",
        "TriggerTime_3PMT_2DOM",
    ]

    tray.AddModule(
        DAQToPhysics,
        "DAQToPhysics",
        keys_to_copy=keys_to_copy,
    )

    # 6) Write ONLY Physics frames (now they contain the stuff we care about)
    tray.AddModule(
        "I3Writer",
        "writer",
        Filename=outfile,
        Streams=[icetray.I3Frame.Physics],
    )

    print("INPUT :", infile)
    print("GCD   :", args.gcdfile)
    print("OUT   :", outfile)
    print("PHOTON:", args.photon_key)
    print("FullDetectorCoincidenceN:", args.nDOMs, "min_time_sep:", args.pulsesep)

    tray.Execute()
    tray.Finish()


if __name__ == "__main__":
    main()
