from graphnet.data.extractors.icecube.utilities.frames import frame_is_montecarlo, frame_is_noise
from typing import Dict, Any, Tuple, OrderedDict, List
from icecube import dataclasses, dataio
from graphnet.data.extractors.icecube import (
    I3TruthExtractor,
    I3Extractor
)
from graphnet.data.dataclasses import I3FileSet



class I3TruthExtractorPONE(I3TruthExtractor):
      
   def _get_primary_particle_interaction_type_and_elasticity(
        self,
        frame: "icetray.I3Frame",
        sim_type: str,
        padding_value: float = -1.0,
    ) -> Tuple[Any, int, float]:
        """Return primary particle, interaction type, and elasticity.

        A case handler that does two things:
            1) Catches issues related to determining the primary MC particle.
            2) Error handles cases where interaction type and elasticity
                doesn't exist

        Args:
            frame: Physics frame containing MC record.
            sim_type: Simulation type.
            padding_value: The value used for padding.

        Returns
            A tuple containing the MCInIcePrimary, if it exists; the primary
                particle, encoded as 1 (charged current), 2 (neutral current),
                or 0 (neither); and the elasticity in the range ]0,1[.
        """
        if sim_type != "noise":
            try:
                MCInIcePrimary = frame["MCInIcePrimary"]
            except KeyError:
                MCInIcePrimary = frame[self._mctree][0]
            if (
                MCInIcePrimary.energy != MCInIcePrimary.energy
            ):  # This is a nan check. Only happens for some muons
                # where second item in MCTree is primary. Weird!
                MCInIcePrimary = frame[self._mctree][1]
                # For some strange reason the second entry is identical in
                # all variables and has no nans (always muon)
        else:
            MCInIcePrimary = None

        if sim_type == "LeptonInjector":
            event_properties = frame["EventProperties"]
            final_state_1 = event_properties.finalType1
            if final_state_1 in [
                dataclasses.I3Particle.NuE,
                dataclasses.I3Particle.NuMu,
                dataclasses.I3Particle.NuTau,
                dataclasses.I3Particle.NuEBar,
                dataclasses.I3Particle.NuMuBar,
                dataclasses.I3Particle.NuTauBar,
            ]:
                interaction_type = 2  # NC
            else:
                interaction_type = 1  # CC

            elasticity = 1 - event_properties.finalStateY

        else:
            try:
                interaction_type = frame["I3MCWeightDict"]["InteractionType"]
            except KeyError:
                interaction_type = int(padding_value)

            try:
                elasticity = 1 - frame["I3MCWeightDict"]["BjorkenY"]
            except KeyError:
                elasticity = padding_value

        return MCInIcePrimary, interaction_type, elasticity
   def __call__(
        self, frame: "icetray.I3Frame", padding_value: Any = -1
    ) -> Dict[str, Any]:
        """Extract truth-level information."""
        is_mc = frame_is_montecarlo(frame, self._mctree)
        is_noise = frame_is_noise(frame, self._mctree)
        sim_type = self._find_data_type(is_mc, self._i3_file, frame)

        output = {
            "energy": padding_value,
            "position_x": padding_value,
            "position_y": padding_value,
            "position_z": padding_value,
            "azimuth": padding_value,
            "zenith": padding_value,
            "pid": padding_value,
            "event_time": frame["I3EventHeader"].start_time.utc_daq_time,
            "sim_type": sim_type,
            "interaction_type": padding_value,
            "elasticity": padding_value,
            "RunID": frame["I3EventHeader"].run_id,
            "SubrunID": frame["I3EventHeader"].sub_run_id,
            "EventID": frame["I3EventHeader"].event_id,
            "SubEventID": frame["I3EventHeader"].sub_event_id,
            "dbang_decay_length": padding_value,
            "track_length": padding_value,
            "stopped_muon": padding_value,
            "energy_track": padding_value,
            "energy_cascade": padding_value,
            "inelasticity": padding_value,
            "DeepCoreFilter_13": padding_value,
            "CascadeFilter_13": padding_value,
            "MuonFilter_13": padding_value,
            "OnlineL2Filter_17": padding_value,
            "L3_oscNext_bool": padding_value,
            "L4_oscNext_bool": padding_value,
            "L5_oscNext_bool": padding_value,
            "L6_oscNext_bool": padding_value,
            "L7_oscNext_bool": padding_value,
            "is_starting": padding_value,
        }


        # here, I removed some stuff about NullSPlit, InIceSplit, Final


        if "FilterMask" in frame:
            if "DeepCoreFilter_13" in frame["FilterMask"]:
                output["DeepCoreFilter_13"] = int(
                    bool(frame["FilterMask"]["DeepCoreFilter_13"])
                )
            if "CascadeFilter_13" in frame["FilterMask"]:
                output["CascadeFilter_13"] = int(
                    bool(frame["FilterMask"]["CascadeFilter_13"])
                )
            if "MuonFilter_13" in frame["FilterMask"]:
                output["MuonFilter_13"] = int(
                    bool(frame["FilterMask"]["MuonFilter_13"])
                )
            if "OnlineL2Filter_17" in frame["FilterMask"]:
                output["OnlineL2Filter_17"] = int(
                    bool(frame["FilterMask"]["OnlineL2Filter_17"])
                )

        elif "DeepCoreFilter_13" in frame:
            output["DeepCoreFilter_13"] = int(bool(frame["DeepCoreFilter_13"]))

        if "L3_oscNext_bool" in frame:
            output["L3_oscNext_bool"] = int(bool(frame["L3_oscNext_bool"]))

        if "L4_oscNext_bool" in frame:
            output["L4_oscNext_bool"] = int(bool(frame["L4_oscNext_bool"]))

        if "L5_oscNext_bool" in frame:
            output["L5_oscNext_bool"] = int(bool(frame["L5_oscNext_bool"]))

        if "L6_oscNext_bool" in frame:
            output["L6_oscNext_bool"] = int(bool(frame["L6_oscNext_bool"]))

        if "L7_oscNext_bool" in frame:
            output["L7_oscNext_bool"] = int(bool(frame["L7_oscNext_bool"]))

        if is_mc and (not is_noise):
            (
                MCInIcePrimary,
                interaction_type,
                elasticity,
            ) = self._get_primary_particle_interaction_type_and_elasticity(
                frame, sim_type
            )

            try:
                (
                    energy_track,
                    energy_cascade,
                    inelasticity,
                ) = self._get_primary_track_energy_and_inelasticity(frame)
            except (
                RuntimeError
            ):  # track energy fails on northeren tracks with ""Hadrons"
                # has no mass implemented. Cannot get total energy."
                energy_track, energy_cascade, inelasticity = (
                    padding_value,
                    padding_value,
                    padding_value,
                )

            output.update(
                {
                    "energy": MCInIcePrimary.energy,
                    "position_x": MCInIcePrimary.pos.x,
                    "position_y": MCInIcePrimary.pos.y,
                    "position_z": MCInIcePrimary.pos.z,
                    "azimuth": MCInIcePrimary.dir.azimuth,
                    "zenith": MCInIcePrimary.dir.zenith,
                    "pid": MCInIcePrimary.pdg_encoding,
                    "interaction_type": interaction_type,
                    "elasticity": elasticity,
                    "dbang_decay_length": self._extract_dbang_decay_length(
                        frame, padding_value
                    ),
                    "energy_track": energy_track,
                    "energy_cascade": energy_cascade,
                    "inelasticity": inelasticity,
                }
            )
            if abs(output["pid"]) == 13:
                output.update(
                    {
                        "track_length": MCInIcePrimary.length,
                    }
                )
                muon_final = self._muon_stopped(output, self._borders)
                output.update(
                    {
                        "position_x": muon_final["x"],
                        # position_xyz has no meaning for muons.
                        # These will now be updated to muon final position,
                        # given track length/azimuth/zenith
                        "position_y": muon_final["y"],
                        "position_z": muon_final["z"],
                        "stopped_muon": muon_final["stopped"],
                    }
                )

            is_starting = self._contained_vertex(output)
            output.update(
                {
                    "is_starting": is_starting,
                }
            )

        return output
   

   def _find_data_type(
        self, mc: bool, input_file: str, frame: "icetray.I3Frame"
    ) -> str:
        """Determine the data type.

        Args:
            mc: Whether `input_file` is Monte Carlo simulation.
            input_file: Path to I3-file.
            frame: Physics frame containing MC record

        Returns:
            The simulation/data type.
        """
        # @TODO: Rewrite to automatically infer `mc` from `input_file`?
        if not mc:
            sim_type = "data"
        elif "muon" in input_file:
            sim_type = "muongun"
        elif "corsika" in input_file:
            sim_type = "corsika"
        elif "genie" in input_file or "nu" in input_file.lower():
            sim_type = "genie"
        elif "noise" in input_file:
            sim_type = "noise"
        elif frame.Has("EventProperties") or frame.Has(
            "LeptonInjectorProperties"
        ):
            sim_type = "LeptonInjector"
        elif frame.Has("I3MCWeightDict"):
            sim_type = "NuGen"
        else:
            raise NotImplementedError("Could not determine data type.")
        return sim_type






















