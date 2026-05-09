from icecube import icetray, dataclasses, simclasses


class FilterFrame(icetray.I3ConditionalModule):
    def __init__(self, ctx):
        super(FilterFrame, self).__init__(ctx)
        self.AddOutBox("OutBox")

        self.AddParameter("AllowedStrings", "List of allowed string IDs", [])
        self.AddParameter("ExcludedOMs", "Optional list of OM IDs to drop within allowed strings", [])
        self.AddParameter("PhotonMapKey", "Photon series map key in DAQ frame", "I3Photons")
        self.AddParameter(
            "OnlyDAQ",
            "If True, do not touch Geometry/Calibration/DetectorStatus frames; only filter DAQ photons",
            True,
        )

    def Configure(self):
        self.photon_map_key = self.GetParameter("PhotonMapKey")
        self.allowed_strings = set(self.GetParameter("AllowedStrings"))
        self.excluded_oms = set(self.GetParameter("ExcludedOMs"))
        self.only_daq = bool(self.GetParameter("OnlyDAQ"))

    def keep_key(self, key):
        if key.string not in self.allowed_strings:
            return False
        om = getattr(key, "om", None)
        return om is None or om not in self.excluded_oms

    def replaceOMGeo(self, frame):
        if "I3OMGeoMap" not in frame or "I3Geometry" not in frame:
            return frame

        omgeo_map = frame["I3OMGeoMap"]
        filtered_omgeo_map = dataclasses.I3OMGeoMap()

        for omkey, omgeo in omgeo_map.items():
            if self.keep_key(omkey):
                filtered_omgeo_map[omkey] = omgeo

        frame.Replace("I3OMGeoMap", filtered_omgeo_map)

        old_geo = frame["I3Geometry"]
        new_geo = dataclasses.I3Geometry()
        new_geo.start_time = old_geo.start_time
        new_geo.end_time = old_geo.end_time
        new_geo.omgeo = filtered_omgeo_map

        frame.Replace("I3Geometry", new_geo)
        return frame

    def replaceModGeo(self, frame):
        if "I3ModuleGeoMap" not in frame:
            return frame

        modgeo_map = frame["I3ModuleGeoMap"]
        filtered_modgeo_map = dataclasses.I3ModuleGeoMap()

        for modkey, modgeo in modgeo_map.items():
            if self.keep_key(modkey):
                filtered_modgeo_map[modkey] = modgeo

        frame.Replace("I3ModuleGeoMap", filtered_modgeo_map)
        return frame

    def replaceSubdet(self, frame):
        if "Subdetectors" not in frame:
            return frame

        subdet = frame["Subdetectors"]
        filtered_det = dataclasses.I3MapModuleKeyString()

        for modkey in subdet.keys():
            if self.keep_key(modkey):
                filtered_det[modkey] = str(subdet[modkey])

        frame.Replace("Subdetectors", filtered_det)
        return frame

    def replaceCal(self, frame):
        if "I3Calibration" not in frame:
            return frame

        ical = frame["I3Calibration"]
        filtered_domcal = dataclasses.Map_OMKey_I3DOMCalibration()

        for omkey, cal in ical.dom_cal.items():
            if self.keep_key(omkey):
                filtered_domcal[omkey] = cal

        newcal = dataclasses.I3Calibration()
        newcal.dom_cal = filtered_domcal
        newcal.start_time = ical.start_time
        newcal.end_time = ical.end_time
        newcal.vem_cal = ical.vem_cal

        frame.Replace("I3Calibration", newcal)
        return frame

    def replaceStatus(self, frame):
        if "I3DetectorStatus" not in frame:
            return frame

        idet = frame["I3DetectorStatus"]
        filtered_domstat = dataclasses.Map_OMKey_I3DOMStatus()

        for omkey, stat in idet.dom_status.items():
            if self.keep_key(omkey):
                filtered_domstat[omkey] = stat

        newdet = dataclasses.I3DetectorStatus()
        newdet.dom_status = filtered_domstat
        newdet.start_time = idet.start_time
        newdet.end_time = idet.end_time
        newdet.daq_configuration_name = idet.daq_configuration_name
        newdet.trigger_status = idet.trigger_status

        frame.Replace("I3DetectorStatus", newdet)
        return frame

    def Geometry(self, frame):
        if self.only_daq:
            self.PushFrame(frame)
            return

        frame = self.replaceOMGeo(frame)
        frame = self.replaceModGeo(frame)
        frame = self.replaceSubdet(frame)
        self.PushFrame(frame)

    def Calibration(self, frame):
        if self.only_daq:
            self.PushFrame(frame)
            return

        frame = self.replaceOMGeo(frame)
        frame = self.replaceModGeo(frame)
        frame = self.replaceSubdet(frame)
        frame = self.replaceCal(frame)
        self.PushFrame(frame)

    def DetectorStatus(self, frame):
        if self.only_daq:
            self.PushFrame(frame)
            return

        frame = self.replaceOMGeo(frame)
        frame = self.replaceModGeo(frame)
        frame = self.replaceSubdet(frame)
        frame = self.replaceCal(frame)
        frame = self.replaceStatus(frame)
        self.PushFrame(frame)

    def DAQ(self, frame):
        if self.photon_map_key not in frame:
            return

        photon_map = frame[self.photon_map_key]
        filtered_map = simclasses.I3CompressedPhotonSeriesMap()

        for module_key, photon_vector in photon_map.items():
            if self.keep_key(module_key):
                filtered_map[module_key] = photon_vector

        if filtered_map:
            frame.Replace(self.photon_map_key, filtered_map)
            self.PushFrame(frame)
