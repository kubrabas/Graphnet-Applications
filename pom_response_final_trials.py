
### burda OM response genel olarak nasil uretiliyor anladim. ama herrrr seyin detayina inmelisin kub daha sonra :)


### elimdeki datada eventler bolunmus ya, gercekte nasil boluncek onlar? ben bos event drop etme olayimda karar vermemi saglayacak sey bu aslinda.

## ToDo: bunlari once bir juypter notebook'ta dene. bazi durumlarda event skip mevzusunu dusun. sorun olcak mi acaba falan diye dusun.
# sonucu da jupyter notebook'ta dene. EventPulseSeries no noise'daki bise normal EventPulseSeries icinde de var mi bak.
## slurm job icin script hazirla. nasildi unuttuysan daha onceki slurm joblarini bulabiliyosundur belki
### belli bir sure sonra hata cikiyosa onlarin ismini kaydetsin. ondaki hata ney ogreneyim.
### skip keys kismina raw kisimda olan seyleri yaz. bosuna bi de onlari kaydetme. noise'lari da bence kaydetmene gerek yok. ama once bi su anki outputta incelemeler yap. asagiy ukari hayalin gibi mi EventPulseSeries'lar. 
### RAM yuksek al knki
## daq ile physics cok farkli mi acaba. ne fark var? :o 
## width ve flags nerde uretilio
## daq kaydetmesem adece physics kaydetsem ok mu?

####### IMPORTS


import os, sys, random
from os.path import expandvars

from DOM.DOMAcceptance import DOMAcceptance
from DOM.PONEDOMLauncher import DOMSimulation

from NoiseGenerators.DarkNoise import DarkNoise
from NoiseGenerators.K40Noise import K40Noise

from icecube.icetray import I3Tray
from icecube import phys_services, sim_services
from icecube import icetray, dataclasses, dataio, simclasses

from PulseCleaning.CausalHits import CausalPulseCleaning
from Trigger.DOMTrigger import DOMTrigger
# from Trigger.DetectorTrigger import DetectorTrigger
# DetectorTrigger yerine:
from icecube.icetray import I3Units, OMKey, I3Frame
from icecube.dataclasses import ModuleKey
import numpy as np
from math import sqrt
from copy import deepcopy


########### my DetectorTrigger:

from icecube import icetray, dataclasses
from icecube.icetray import OMKey
from math import sqrt


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




####### GLOBAL PARAMETERS

runnumber = 1 ## tam anlamadim 
# "The run/dataset number for this simulation, is used as seed for random generator"

dropstrings = []

# nerelerde bu kullanilcak nerelerde baska random lazin
randomService = phys_services.I3SPRNGRandomService(
    seed=1234567, nstreams=10000, streamnum=runnumber
)


infile = "/project/6008051/pone_simulation/MC10-000002-nu_mu-2_7-LeptonInjector-PROPOSAL-clsim/Photon/cls_4910.i3"
# "Read input from INFILE (.i3{.gz} format)"

outfile = '/project/def-nahee/kbas/POM_Response_GZ_new/deneme.i3.gz'
# "Write output to OUTFILE (.i3{.gz} format)"

gcdfile = "/project/6008051/pone_simulation/GCD_Library/PONE_800mGrid.i3.gz"
# "Read in GCD file"




## sample time ne senin datanda
pulsesep = 0.2
# "Time needed to separate two pulses. Assume that this is 3.5*sample time."






######## Additional Classes



## PMT_Response_nonoise içinde pulse üreten PMT key sayısı 5’ten azsa event’i drop ediyor.. aciklamalarda number of OM demis ama bu dogru degil :) number of PMT demeliydi
## acaba aciklama mi yanlis kod mu... acaba kodu degistirsem mi HitCountCheck icin. 5 OM baksin 5 PMT degil
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




######## Pipeline

tray = I3Tray()



tray.context["I3RandomService"] = randomService



tray.AddModule("I3Reader", "reader", FilenameList=[gcdfile, infile]) # boyle ok mu acaba FilenameList
# gpt says : FilenameList=[...] verdiğinde listeyi baştan sona gezer: önce gcdfile biter, sonra otomatik infile’a geçer. (İstersen 10 tane dosya da koyarsın.)
# FilenameList=[gcdfile, infile1, infile2, infile3]


### bu kisim tamamdirrrrrr yayyy
tray.AddModule(DOMAcceptance,
               'DOMAcceptance',
               input_map      = 'I3Photons', 
               output_map     = 'Accepted_PulseMap',
               random_service = randomService,
               # drop_strings : default
               drop_empty = False ## I3Photons eger bossa drop etmesi icin. ben simdilik drop etmiyorum ama  gercek datayi eventlere nasil boluyorlar, buna bagli olmali benim karar veris
               )



##### bu kisim da tamam
tray.AddModule(DarkNoise,
               'AddDarkNoise',
               input_map      = 'Accepted_PulseMap',
               output_map     = 'Noise_Dark',
               # dark_rate : default
               # drop_strings : default
               # drop_oms : default 
               # use_manual_noise_bounds : default
               # manual_noise_bounds : default
               # noise_padding : default
               random_service = randomService,
               gcd_file       = gcdfile
               )
# dark noise, input_map’te (Accepted_PulseMap) zaten pulse olan PMT’lerle sınırlı değil; input_map sadece gürültünün üretileceği zaman aralığını belirlemek için kullanılıyor. Noise, drop_strings/drop_oms ile elenenler hariç, geometrideki (omkeys_to_use) tüm OM’ler üzerinde dönülerek, POM modelinin tanımladığı tüm PMT indeksleri (1..num_pmts) için rastgele (Poisson) hit’ler üretilerek ekleniyor; ancak Poisson olduğu için her PMT’de mutlaka hit oluşmayabilir.



##### bu kisim da tamam
tray.AddModule(K40Noise,
               'AddK40Noise',
               input_map      = 'Accepted_PulseMap',
               output_map     = 'Noise_K40', 
               # characterization_file : default
               random_service = randomService,
               # drop_strings : default
               # drop_oms : default 
               # use_manual_noise_bounds : default 
               # manual_noise_bounds : default
               # noise_padding : default
               gcd_file       =  gcdfile
               )
# K40Noise için de mantık aynı temel prensibe dayanıyor: **`input_map` hangi PMT’lere noise ekleneceğini belirlemiyor; sadece noise üretilecek zaman penceresini belirlemek için kullanılıyor.** Noise, `drop_strings/drop_oms` ile elenmeyen **tüm OM’ler** üzerinde üretiliyor. Ancak DarkNoise’tan farkı şu: DarkNoise her OM’de *her PMT için bağımsız Poisson hit* üretebilirken, **K40Noise her OM içinde K40 “event/correlation” üretip bu eventlerin vuracağı PMT’leri rastgele seçiyor** (single-fold ise tek PMT; multi-fold ise karakterizasyon dosyasından gelen PMT kombinasyonları + flip/rotasyon ile modül üzerindeki PMT’lere dağıtılıyor). Dolayısıyla **teoride tüm PMT’ler hit alabilir**, ama pratikte **her eventte sadece seçilen PMT’lere** hit yazılıyor; hit almayan PMT’ler map’e hiç girmiyor.


# noise uretimleri icin:
# alt sınır = (input_map’teki en erken zaman) − 2000 ns
# üst sınır = (input_map’teki en geç zaman) + 10000 ns


# küb.. defaulttan kastim Examples dosyasi degil. her bir class'in kendi source code'u


## kübbb bu noise uretme kisimlarinda source kodlari nasil calisiyor ogren cnm


####### bu da tamam ama drop empty kismini bir dusun
tray.AddModule(DOMSimulation,   ### bunun kullanimi source kodda yanlis ya. input map icin I3Photons kullanilmis
               'DOMLauncher',
               input_map      = 'Accepted_PulseMap',
               output_map     = 'PMT_Response',    ## no noise'lisi da uretilecek
               # outputmap_mcpe : default (in the source codes, it is mentioned that this parameter is not used)
               # pmt_tts : default
               # pmt_ts : default
               # charge_sigma : default
               # charge_mean : default
               # afterpulse_prob : default
               # afterpulse_meantime_1 : default
               # afterpulse_timesigma_1 : default
               # afterpulse_meantime_2 : default
               # afterpulse_timesigma_2 : default
               # afterpulse_componet_ratio : default
               # late_pulse_prob : default
               # pe_threshold : default
               # pe_saturation : default
               # drop_strings : default
               # noise_pulse_series : default
               no_pure_noise_events = False,
               drop_empty = True,
               random_service = randomService,
               min_time_sep   = pulsesep,
               split_doms     = True,
               use_dark       = True,
               dark_map       = 'Noise_Dark',
               use_k40        = True,
               k40_map        = 'Noise_K40'
              )


# `drop_empty` ve `no_pure_noise_events` kontrolleri, **PMT_Response ** ve **PMT_Response_nonoise 
# daha üretilmeden önce**, DAQ fonksiyonunun başında çalışıyor; yani drop kararı sadece `Accepted_PulseMap` 
# (gerçek/simülasyon sinyali) ve varsa `DarkHits`/`K40Hits` (noise map’leri) üzerinden veriliyor. 
# `drop_empty=True` ve `no_pure_noise_events=False` demek: **sinyal yoksa bile eğer noise varsa event’i tut**, 
# ama **ne sinyal ne de noise varsa tamamen boş event’i drop et** anlamına geliyor. yani event komplr drop ediliyor.

# bu da tm ama class'ta degisiklik yapmak isteyebilirsin.
#`HitCountCheck`, `PMT_Response_nonoise` içinde pulse üreten PMT sayısı 5’ten az olan event’leri düşürür, 5 ve üzeri olanları geçirir.
tray.AddModule(HitCountCheck, "hitcheck", NHits=5)


tray.AddModule(
    DOMTrigger,
    "DOMTrigger",
    trigger_map="triggerpulsemap",
    # output : default
    # inputmap : default
    # PEthreshold : default
    CutNotTriggered = True, # DOMCoincidence_ncoin bos olursa o frame komple dusurulur. dusurulmese ne olcakti. (yani eger herhangi bir domda bile ayni anda (yakin aralikta) 3 pmt hit almadiysa, dusurulur)
    # SingleDOMCoincidenceN : default
    # SingleDOMCoincidenceWindow : default
    # SingleStringNRows : default
    # ForceAdjacency : default
)

## bunun anlamini daha iyi ogren.
nDOMs = 2. # bu 1'di ben _3PMT_2DOM isminden dolayi degistirdim.
# "Number of DOMs for detector trigger"

tray.AddModule(
    DetectorTrigger,
    "PONE_Trigger",
    output="_3PMT_2DOM",
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
    CutOnTrigger=True, # detector trigger, string trigger , single-OM trigger eger bos ise drop edilir o event
    EventLength=10000,
    TriggerTime=2000)




tray.AddModule(
    "I3Writer",
    "writer",
    SkipKeys = ["I3Photons",""],
    Filename=outfile,
    Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
)


tray.Execute()
tray.Finish()


### bu daq'ta EventHeader vsvs hepsi var mi?
### sadece physics denedim:
### sadece daq denedim: 
###