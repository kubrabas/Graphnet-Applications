import glob
import os, sys, random
from os.path import expandvars

from pathlib import Path

# --- make pone_offline imports work although this script lives outside pone_offline ---
DEFAULT_PONE_ENV_DIR = "/project/def-nahee/kbas/pone_offline"
PONE_ENV_DIR = os.environ.get("PONE_ENV_DIR", DEFAULT_PONE_ENV_DIR)
if PONE_ENV_DIR not in sys.path:
    sys.path.insert(0, PONE_ENV_DIR)

from DOM.DOMAcceptance import DOMAcceptance
from DOM.PONEDOMLauncher import DOMSimulation

from NoiseGenerators.DarkNoise import DarkNoise
from NoiseGenerators.K40Noise import K40Noise

from icecube.icetray import I3Tray
from icecube import phys_services, sim_services
from icecube import icetray, dataclasses, dataio, simclasses

from PulseCleaning.CausalHits import CausalPulseCleaning
from Trigger.DOMTrigger import DOMTrigger



import os, sys, random, re
from os.path import expandvars

from pathlib import Path

from DOM.DOMAcceptance import DOMAcceptance
from DOM.PONEDOMLauncher import DOMSimulation

from NoiseGenerators.DarkNoise import DarkNoise
from NoiseGenerators.K40Noise import K40Noise

from icecube.icetray import I3Tray, I3Units, OMKey, I3Frame
from icecube import phys_services, sim_services
from icecube import icetray, dataclasses, dataio, simclasses

from PulseCleaning.CausalHits import CausalPulseCleaning
from Trigger.DOMTrigger import DOMTrigger

# I dont use:
# from Trigger.DetectorTrigger import DetectorTrigger
# but rather, I have my own DetectorTrigger

from icecube.dataclasses import ModuleKey
import numpy as np
from math import sqrt
from copy import deepcopy
import argparse


# Custom classes are kept as they inherit from I3Module directly.
print("script check point 2")

# --- PATHS AND SLURM CONFIGURATION ---

try:
    SLURM_ARRAY_TASK_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
except KeyError:
    # Use 0 for manual testing if not run in Slurm
    SLURM_ARRAY_TASK_ID = 0
print("script check point 3")

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, required=True)
args = parser.parse_args()
file_path = args.infile

# --- runtime configuration from env / slurm wrapper ---
FLAVOR = os.environ["FLAVOR"]
SUB_GEOMETRY = os.environ["SUB_GEOMETRY"]
SIM_PATH = os.environ["SIM_PATH"]
OUTPUT_FOLDER = os.environ["OUTPUT_FOLDER"]
GCD_FILE = os.environ["GCD_FILE"]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print("script check point 4")

print(f"--- CONFIG: flavor={FLAVOR}")
print(f"--- CONFIG: sub_geometry={SUB_GEOMETRY}")
print(f"--- CONFIG: sim_path={SIM_PATH}")
print(f"--- CONFIG: output_folder={OUTPUT_FOLDER}")
print(f"--- CONFIG: gcd_file={GCD_FILE}")
print(f"--- CONFIG: pone_env_dir={PONE_ENV_DIR}")

# --- SELECT THE FILE FOR THIS ARRAY JOB ---
if not os.path.exists(file_path):
    print(f"Input file does not exist: {file_path}")
    sys.exit(1)
print("script check point 5")

# --- FILE NAMING ---
base_name = os.path.basename(file_path)

m = re.search(r"(?:gen|cls)_(\d+)", base_name)
if not m:
    raise ValueError(f"Could not parse run number from filename: {base_name}")

file_id_part = m.group(1)


name_no_ext = base_name
if name_no_ext.endswith(".i3.gz"):
    name_no_ext = name_no_ext[:-6]
elif name_no_ext.endswith(".i3.zst"):
    name_no_ext = name_no_ext[:-7]
elif name_no_ext.endswith(".i3"):
    name_no_ext = name_no_ext[:-3]

if name_no_ext.endswith("_skim"):
    new_filename = f"{name_no_ext[:-5]}_pmt_response.i3.gz"
else:
    new_filename = f"{name_no_ext}_pmt_response.i3.gz"

output_path = os.path.join(OUTPUT_FOLDER, new_filename)

if os.path.exists(output_path):
    print(f"Output already exists: {output_path} (skipping)")
    sys.exit(0)



print(f"Starting Array Job {SLURM_ARRAY_TASK_ID}. Processing {file_path}")
print(f"--- CONFIG: infile={file_path}")
print(f"--- CONFIG: output_path={output_path}")
print("script check point 6")

# --- TRAY CONFIGURATION ---
runnumber = int(file_id_part) # "The run/dataset number for this simulation, is used as seed for random generator"
pulsesep = 0.2 # "Time needed to separate two pulses. Assume that this is 3.5*sample time." what is sample time in this data?

nDOMs = 1 # "Number of DOMs for detector trigger"
# I changed the name _3PMT_2DOM to _3PMT_1DOM 

randomService = phys_services.I3SPRNGRandomService(
    seed=1234567, nstreams=10000, streamnum=runnumber
)
# Understand randomService
print("script check point 7")


####### here is my own DetectorTrigger:
class DetectorTrigger(icetray.I3ConditionalModule):
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("output", "Append the outputs", "")
        self.AddParameter("input", "Name of the Physics I3MCTree name", "")
        self.AddParameter("CutOnTrigger", "Cut events that do not trigger.", False)
        self.AddParameter("FullDetectorCoincidenceN", "", 3)
        self.AddParameter("FullDetectorCoincidenceWindow", "", 1.1)
        self.AddParameter("StringCoincidenceN", "", 2)
        self.AddParameter("StringCoincidenceWindow", "", 1.1)
        self.AddParameter("StringNRows", "", 3)
        self.AddParameter("StringDist", "", 1.5)
        self.AddParameter("ScaleBySpacing", "Turn the Windows inputs to be relative to detector size.", True)
        self.AddParameter("ForceAdjacency", "Require adjacency ", True)
        self.AddParameter("OMPMTCoinc", "Number of PMTs needed for OM Trigger", 2)
        self.AddParameter("EventLength", "Length of Event", 10000)
        self.AddParameter("TriggerTime", "Time of trigger in event.", 2000)
        self.AddParameter("PulseSeriesIn", "Pulse series in", "PMT_Response")
        self.AddParameter("PulseSeriesOut", "Pulse series out", "EventPulseSeries")
        self.AddParameter("PulseSeriesInNoNoise", "No-noise pulse series in", "PMT_Response_nonoise")
        self.AddParameter("PulseSeriesOutNoNoise", "No-noise pulse series out", "EventPulseSeries_nonoise")
        self.AddParameter("SingleOMTriggerCoince", " ", 3)
        self.AddOutBox("OutBox")

    def Configure(self):
        self.output = self.GetParameter("output")
        self.input = self.GetParameter("input")
        self.CutOnTrigger = self.GetParameter("CutOnTrigger")
        self.FullDetectorCoincidenceN = self.GetParameter("FullDetectorCoincidenceN")
        self.StringCoincidenceN = self.GetParameter("StringCoincidenceN")
        self.ForceAdjacency = self.GetParameter("ForceAdjacency")
        self.StringNRows = self.GetParameter("StringNRows")
        self.StringDist = self.GetParameter("StringDist")
        self.FullDetectorCoincidenceWindow_unscaled = self.GetParameter("FullDetectorCoincidenceWindow")
        self.StringCoincidenceWindow_unscaled = self.GetParameter("StringCoincidenceWindow")
        self.OMPMTCoinc = self.GetParameter("OMPMTCoinc")
        self.EventLength = self.GetParameter("EventLength")
        self.TriggerTime = self.GetParameter("TriggerTime")
        self.PulseSeriesIn = self.GetParameter("PulseSeriesIn")
        self.PulseSeriesOut = self.GetParameter("PulseSeriesOut")
        self.PulseSeriesInNoNoise = self.GetParameter("PulseSeriesInNoNoise")
        self.PulseSeriesOutNoNoise = self.GetParameter("PulseSeriesOutNoNoise")
        self.ScaleBySpacing = self.GetParameter("ScaleBySpacing")
        self.DoStringTrigger = True
        self.DoCoincTriggers = True
        self.SingleOMPMTTriggerCoince = self.GetParameter("SingleOMTriggerCoince")
        if self.SingleOMPMTTriggerCoince <= self.OMPMTCoinc:
            self.DoCoincTriggers = False

        self.nstrings = int(0)
        self.nOMs = int(0)
        self.npmts = int(0)
        self.has_seen_geometry = False
        self.eventcount = 0

    def Geometry(self, frame):
        self.has_seen_geometry = True

        if self.ScaleBySpacing:
            geo = frame["I3Geometry"].omgeo

            stringlist = set()
            omlist = set()
            pmtlist = set()
            for omkey in geo.keys():
                stringlist.add(omkey.string)
                omlist.add(omkey.om)
                pmtlist.add(omkey.pmt)

            stringlist = sorted(stringlist)
            self.nstrings = len(stringlist)
            self.nOMs = len(omlist)
            self.npmts = len(pmtlist)

            keys = list(geo.keys())
            OM_space = abs(geo[keys[0]].position.z - geo[keys[self.npmts]].position.z)
            string_pos = []
            for string in stringlist:
                string_pos.append(geo[OMKey(string, 1, 1)].position)

            average_min_stringdist = 0.0
            for i in range(len(string_pos) - 1):
                this_min_stringdist = 99999.0
                omposi = string_pos[i]
                for j in range(i + 1, len(string_pos)):
                    omposj = string_pos[j]
                    dist = sqrt((omposi.x - omposj.x) ** 2.0 + (omposi.y - omposj.y) ** 2.0)
                    if dist < this_min_stringdist:
                        this_min_stringdist = dist
                average_min_stringdist += this_min_stringdist
            average_min_stringdist /= (self.nstrings - 1)

            maxOMDistance = 0.0
            om_pos = []
            for omkey in geo.keys():
                if (omkey.om == 1 or omkey.om == self.nOMs) and (omkey.pmt == 1):
                    om_pos.append(geo[omkey].position)

            for i in range(len(om_pos) - 1):
                omposi = om_pos[i]
                for j in range(i + 1, len(om_pos)):
                    omposj = om_pos[j]
                    dist = sqrt(
                        (omposi.x - omposj.x) ** 2.0
                        + (omposi.y - omposj.y) ** 2.0
                        + (omposi.z - omposj.z) ** 2.0
                    )
                    if dist > maxOMDistance:
                        maxOMDistance = dist

            self.FullDetectorCoincidenceWindow = (
                self.FullDetectorCoincidenceWindow_unscaled * maxOMDistance / 0.3
                + OM_space * 1.3 / 0.3
            )
            self.StringCoincidenceWindow = (
                self.StringCoincidenceWindow_unscaled * average_min_stringdist / 0.3
                + OM_space * 1.3 / 0.3
            )
        else:
            self.FullDetectorCoincidenceWindow = self.FullDetectorCoincidenceWindow_unscaled
            self.StringCoincidenceWindow = self.StringCoincidenceWindow_unscaled

        if self.StringCoincidenceN >= self.FullDetectorCoincidenceN:
            if self.StringCoincidenceWindow <= self.FullDetectorCoincidenceWindow:
                self.DoStringTrigger = False

        self.StringTriggerGroups = []

        if self.ForceAdjacency:
            nstart = int((self.StringNRows - 1) / 2)
            for j in range(self.nstrings):
                for i in range(nstart, self.nOMs - nstart):
                    self.StringTriggerGroups.append([])
                    for l in range(len(string_pos)):
                        if (
                            sqrt(
                                (string_pos[j].x - string_pos[l].x) ** 2.0
                                + (string_pos[j].y - string_pos[l].y) ** 2.0
                            )
                            < average_min_stringdist * 1.5
                        ):
                            for k in range(i - nstart, i + nstart + 1):
                                self.StringTriggerGroups[-1].append(OMKey(stringlist[l], k + 1, 1))
        else:
            nstart = int((self.StringNRows - 1) / 2)
            for i in range(nstart, self.nOMs - nstart):
                self.StringTriggerGroups.append([])
                for j in range(self.nstrings):
                    for k in range(i - nstart, i + nstart + 1):
                        self.StringTriggerGroups[-1].append(OMKey(stringlist[j], k, 1))

        self.OMTriggerGroups = {}
        for i in range(len(self.StringTriggerGroups)):
            for om in self.StringTriggerGroups[i]:
                if om not in self.OMTriggerGroups:
                    self.OMTriggerGroups[om] = []
                self.OMTriggerGroups[om].append(i)

        self.PushFrame(frame)

    def DetectorStatus(self, frame):
        if not self.has_seen_geometry:
            raise RuntimeError("This module needs a Geometry frame in your input stream")

        frame["FullDetectorCoincidenceWindow" + self.output] = dataclasses.I3Double(self.FullDetectorCoincidenceWindow)
        frame["StringCoincidenceWindow" + self.output] = dataclasses.I3Double(self.StringCoincidenceWindow)
        frame["FullDetectorCoincidenceN" + self.output] = dataclasses.I3Double(self.FullDetectorCoincidenceN)
        frame["StringCoincidenceN" + self.output] = dataclasses.I3Double(self.StringCoincidenceN)
        frame["TriggerForceAdjacency" + self.output] = dataclasses.I3Double(self.ForceAdjacency)

        self.PushFrame(frame)

    def GetOMTriggers(self, OMCoincidence_time, OMCoincidence_ncoin, OMCoincidence_pmts, ncoincidence):
        OMTriggers = {}
        for key in OMCoincidence_time.keys():
            for i in range(len(OMCoincidence_time[key])):
                time = OMCoincidence_time[key][i]
                coinc = OMCoincidence_ncoin[key][i]
                if coinc >= ncoincidence:
                    if key not in OMTriggers:
                        OMTriggers[key] = []
                    OMTriggers[key].append(time)
        return OMTriggers

    def DAQ(self, frame):
        OMCoincidence_time = frame["DOMTrigger_Time" + self.input]
        OMCoincidence_ncoin = frame["DOMTrigger_NCoin" + self.input]
        OMCoincidence_pmts = frame["DOMTrigger_PMTs" + self.input]

        stringTriggerTime = dataclasses.I3VectorDouble()
        detectorTriggerTime = dataclasses.I3VectorDouble()
        singleOMTriggerTime = dataclasses.I3VectorDouble()

        SingleOMTriggers = self.GetOMTriggers(OMCoincidence_time, OMCoincidence_ncoin, OMCoincidence_pmts, 3)

        if self.DoCoincTriggers:
            OMTriggers = self.GetOMTriggers(
                OMCoincidence_time, OMCoincidence_ncoin, OMCoincidence_pmts, self.OMPMTCoinc
            )

            FullDetectOMTriggers = []
            StringTriggers = {}

            for key in OMTriggers.keys():
                for i in self.OMTriggerGroups[key]:
                    if i not in StringTriggers:
                        StringTriggers[i] = []
                    for time in OMTriggers[key]:
                        StringTriggers[i].append((key, time))
                for time in OMTriggers[key]:
                    FullDetectOMTriggers.append((key, time))

            StringTrigOpp = {}
            if self.DoStringTrigger:
                for i in StringTriggers.keys():
                    StringTrigOpp[i] = []
                    for j in range(len(StringTriggers[i])):
                        StringTrigOpp[i].append([StringTriggers[i][j][1], [StringTriggers[i][j][0]], [StringTriggers[i][j][1]]])
                    for k in range(len(StringTrigOpp[i])):
                        for j in range(len(StringTriggers[i])):
                            if (
                                abs(StringTriggers[i][j][1] - StringTrigOpp[i][k][0]) < self.StringCoincidenceWindow
                                and StringTriggers[i][j][0] not in StringTrigOpp[i][k][1]
                            ):
                                StringTrigOpp[i][k][1].append(StringTriggers[i][j][0])
                                StringTrigOpp[i][k][2].append(StringTriggers[i][j][1])

            DetectTrigOpp = []
            for j in range(len(FullDetectOMTriggers)):
                DetectTrigOpp.append([FullDetectOMTriggers[j][1], [FullDetectOMTriggers[j][0]], [FullDetectOMTriggers[j][1]]])

            for k in range(len(DetectTrigOpp)):
                for j in range(len(FullDetectOMTriggers)):
                    if (
                        (FullDetectOMTriggers[j][1] - DetectTrigOpp[k][0]) < self.FullDetectorCoincidenceWindow
                        and (FullDetectOMTriggers[j][1] - DetectTrigOpp[k][0]) >= 0.0
                        and (FullDetectOMTriggers[j][0] not in DetectTrigOpp[k][1])
                    ):
                        DetectTrigOpp[k][1].append(FullDetectOMTriggers[j][0])
                        DetectTrigOpp[k][2].append(FullDetectOMTriggers[j][1])

            for i in range(len(DetectTrigOpp)):
                if len(DetectTrigOpp[i][1]) >= self.FullDetectorCoincidenceN:
                    triggered = False
                    t_candidate = max(DetectTrigOpp[i][2])
                    for k in range(len(detectorTriggerTime)):
                        if abs(detectorTriggerTime[k] - t_candidate) < self.EventLength:
                            detectorTriggerTime[k] = min(detectorTriggerTime[k], t_candidate)
                            triggered = True
                    if not triggered:
                        detectorTriggerTime.append(t_candidate)

            if self.DoStringTrigger:
                for j in StringTrigOpp.keys():
                    for i in range(len(StringTrigOpp[j])):
                        if len(StringTrigOpp[j][i][1]) >= self.StringCoincidenceN:
                            triggered = False
                            t_candidate = max(StringTrigOpp[j][i][2])
                            for k in range(len(stringTriggerTime)):
                                if abs(stringTriggerTime[k] - t_candidate) < self.EventLength:
                                    stringTriggerTime[k] = min(stringTriggerTime[k], t_candidate)
                                    triggered = True
                            if not triggered:
                                stringTriggerTime.append(t_candidate)

        for om in SingleOMTriggers.keys():
            for time in SingleOMTriggers[om]:
                singleOMTriggerTime.append(time)

        if (
            self.CutOnTrigger
            and len(stringTriggerTime) < 1
            and len(detectorTriggerTime) < 1
            and len(singleOMTriggerTime) < 1
        ):
            return

        frame["DetectorTriggers" + self.output] = detectorTriggerTime
        frame["StringTriggers" + self.output] = stringTriggerTime
        frame["singleOMTrigger" + self.output] = singleOMTriggerTime

        pulseseriesmap = frame[self.PulseSeriesIn]
        outputpulsemap = dataclasses.I3RecoPulseSeriesMap()

        mintrigtime = 99999999.0
        for _time in detectorTriggerTime:
            if _time < mintrigtime:
                mintrigtime = _time
        for _time in stringTriggerTime:
            if _time < mintrigtime:
                mintrigtime = _time
        for _time in singleOMTriggerTime:
            if _time < mintrigtime:
                mintrigtime = _time

        tmin = mintrigtime - self.TriggerTime
        tmax = mintrigtime + self.EventLength - self.TriggerTime



        for om in pulseseriesmap.keys():
            pulseseries = dataclasses.I3RecoPulseSeries()
            for pulse in pulseseriesmap[om]:
                if pulse.time > tmin and pulse.time < tmax:
                    resetpulse = dataclasses.I3RecoPulse()
                    resetpulse.charge = pulse.charge
                    resetpulse.time = pulse.time - mintrigtime + self.TriggerTime
                    pulseseries.append(resetpulse)
            if len(pulseseries) > 0:
                outputpulsemap[om] = pulseseries

        frame[self.PulseSeriesOut] = outputpulsemap
        frame["TriggerTime" + self.output] = dataclasses.I3Double(mintrigtime)

        pulseseriesmap_nn = frame[self.PulseSeriesInNoNoise]
        outputpulsemap_nn = dataclasses.I3RecoPulseSeriesMap()
        for om in pulseseriesmap_nn.keys():
            pulseseries_nn = dataclasses.I3RecoPulseSeries()
            for pulse in pulseseriesmap_nn[om]:
                if pulse.time > tmin and pulse.time < tmax:
                    resetpulse = dataclasses.I3RecoPulse()
                    resetpulse.charge = pulse.charge
                    resetpulse.time = pulse.time - mintrigtime + self.TriggerTime
                    pulseseries_nn.append(resetpulse)
            if len(pulseseries_nn) > 0:
                outputpulsemap_nn[om] = pulseseries_nn
        frame[self.PulseSeriesOutNoNoise] = outputpulsemap_nn

        self.PushFrame(frame)
        Pframe = icetray.I3Frame("P")
        self.PushFrame(Pframe)
## what I have added in this new DetectorTrigger:
### once event trigger time is calculated using triggerpulsemap and the outcomes from DOMTrigger, PMT_Response and PMT_Response_nonoise has the same window cut and same t_new adjustment 




# do I want to use this:
class HitCountCheck(icetray.I3Module):
    def __init__(self, context):
        super(HitCountCheck, self).__init__(context)
        self.AddParameter("NHits", "required number of OMs hit to pass frame","") 
    def Configure(self):
        self.NHits=self.GetParameter("NHits")

    def DAQ(self,frame):
        if(len(frame['PMT_Response_nonoise'])<self.NHits):
            return False
        else:
            self.PushFrame(frame)
## NOTE: I took this from Examples/POM_Response. it says "required number of OMs hit to pass frame" but it actually checks number of PMT's. if i change PMT_Response_nonoise to triggerpulsemap, maybe it could check number of OM's? or maybe I dont need to use this class at all? I already have triggering system? but still I used it here.




# Set up the IceTray pipeline
tray = I3Tray()
photon_series = "I3Photons"
tray.context["I3RandomService"] = randomService
tray.AddModule("I3Reader", "reader", FilenameList=[GCD_FILE, file_path])


print("script check point 8")

# Modules are added by STRING NAME to avoid Python import conflicts
tray.AddModule(DOMAcceptance, 'DOMAcceptance', 
               input_map=photon_series, output_map='Accepted_PulseMap', 
               random_service=randomService,
               drop_empty=True)

## drop_empty: drops the event if I3Photons is empty. I dont want to drop, because even if there is no any real pulse coming to the OM, still there will be noises. But I dropped here.
## drop_empty do not drop the cases where OM receives photons (I3Photons is not empty empty) but they are not accepted by any PMT, they will still be included. It only drops frames where I3Photons are empty


tray.AddModule(DarkNoise, 'AddDarkNoise', 
               input_map='Accepted_PulseMap', output_map='Noise_Dark', 
               random_service=randomService, gcd_file=GCD_FILE)

# Dark noise is not limited to PMTs that already have pulses in the input_map (Accepted_PulseMap); the input_map is used only to define the time window over which the noise will be generated. Noise is added by iterating over all OMs in the geometry (omkeys_to_use), excluding those removed via drop_strings/drop_oms, and generating random (Poisson) hits for all PMT indices (1..num_pmts) defined by the POM model. However, since the process is Poisson, a hit is not guaranteed to occur on every PMT.



tray.AddModule(K40Noise, 'AddK40Noise', 
               input_map='Accepted_PulseMap', output_map='Noise_K40', 
               random_service=randomService, gcd_file=GCD_FILE)

# K40Noise follows the same basic principle: `input_map` does not determine which PMTs will receive noise; it is only used to define the time window over which noise is generated. Noise is produced for all OMs that are not excluded via `drop_strings/drop_oms`. The difference from DarkNoise is this: while DarkNoise can generate independent Poisson hits for each PMT in each OM, K40Noise generates K40 “events/correlations” within each OM and randomly selects which PMTs those events will hit (for single-fold, a single PMT; for multi-fold, PMT combinations taken from the characterization file, then mapped onto the module’s PMTs via flip/rotation). Therefore, in theory all PMTs can receive hits, but in practice each event writes hits only to the selected PMTs.


# If a manual boundary is not selected, the boundaries for noise generation for each event are:
## lower bound = (earliest time in the input_map) − 2000 ns
## upper bound = (latest time in the input_map) + 10000 ns



tray.AddModule(DOMSimulation, 'DOMLauncher', 
               input_map='Accepted_PulseMap', 
               output_map='PMT_Response', # PMT_Response_nonoise will also be produced
               random_service=randomService, min_time_sep=pulsesep, split_doms=True, 
               use_dark=True, dark_map='Noise_Dark', use_k40=True, k40_map='Noise_K40')
# The `drop_empty` and `no_pure_noise_events` checks run at the beginning of the DAQ function, before `PMT_Response` and `PMT_Response_nonoise` are produced. In other words, the drop decision is made based only on `Accepted_PulseMap` (the true/simulated signal) and, if present, the noise maps `DarkHits`/`K40Hits`.
# understand all parameters of this class


# Filter
tray.AddModule(HitCountCheck, "hitcheck", NHits=5)
print("script check point 9")

tray.AddModule(DOMTrigger, "DOMTrigger", trigger_map="triggerpulsemap")
# triggerpulsemap is POMResponse including noise and pulse. it is on OM level.
# understand all parameters of this class



tray.AddModule(
    DetectorTrigger,
    "PONE_Trigger",
    output="_3PMT_1DOM",
    # input : default
    # FullDetectorCoincidenceWindow : default
    # StringCoincidenceN : default
    # StringCoincidenceWindow : default
    # StringNRows : default
    # StringDist : default
    # ScaleBySpacing : default
    # ForceAdjacency : default
    # PulseSeriesIn : default (PMT_Response)
    # PulseSeriesOut : default (EventPulseSeries)
    # PulseSeriesInNoNoise : default (PMT_Response_nonoise)
    # PulseSeriesOutNoNoise : default (EventPulseSeries_nonoise)
    # SingleOMTriggerCoince : default
    OMPMTCoinc=3,
    FullDetectorCoincidenceN=nDOMs,
    CutOnTrigger=True, # if all "detector trigger", "string trigger" and, "single-OM trigger" are empty, then the event is dropped. 
    EventLength=10000,
    TriggerTime=2000)
# I believe it makes sense to drop  non-triggered (?) events. Because in the real experiment setup, they are not recorded?
# is my definition of event being triggered correct? I believe, if TriggerTime_3PMT_2DOM could be calculated (it is not 99999999.0), then the event was triggered? so I set CutOnTrigger=True, but in the Example/POM_Response, it was set CutOnTrigger=False
# understand all parameters of this class






# Write results
tray.AddModule(
    "I3Writer",
    "writer",
    Filename=output_path,
    Streams=[icetray.I3Frame.DAQ],  ## I checked results for both DAQ and Physics frame. They looked pretty same, so here I created only DAQ. Is this correct?
    SkipKeys = ["I3Photons", "Accepted_PulseMap",  
                    "Noise_Dark", "Noise_K40", 
                    "PMT_Response", "PMT_Response_nonoise", "triggerpulsemap"]
)
print("script check point 10")

# Execute the simulation
tray.Execute()
tray.Finish()

print(f" Finished Array Job {SLURM_ARRAY_TASK_ID}. Output: {output_path}")






## Notes
# In this script, reproducible noise generation is ensured by using a fixed random seed (e.g., `1234567`) and assigning a unique `streamnum` to each input file, typically derived from the file’s batch ID parsed from the filename. This way, rerunning the same infile produces identical noise, while different infiles get different random streams. To make this work reliably, `nstreams` is set sufficiently large so that it covers the full range of possible `streamnum` values (for example, if batch IDs can be a few thousand like 4910, `nstreams` is chosen as `100000` or larger).