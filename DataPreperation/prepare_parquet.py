from __future__ import annotations

"""PONE I3 -> Parquet conversion script with custom reader and extractors."""

# ============================================================
# Standard library imports
# ============================================================

import glob
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, OrderedDict

# ============================================================
# Third-party imports
# ============================================================

import matplotlib.path as mpath
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

# ============================================================
# GraphNeT / IceCube availability check
# ============================================================

from graphnet.utilities.imports import has_icecube_package

if has_icecube_package():
    from icecube import icetray  # noqa: F401
else:
    raise RuntimeError("IceCube/IceTray environment not available.")

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (  # noqa: F401
        dataclasses,
        dataio,
        icetray,
        phys_services,
        LeptonInjector,
    )  # pyright: reportMissingImports=false

# ============================================================
# GraphNeT imports
# ============================================================

from graphnet.data.dataclasses import I3FileSet
from graphnet.data.dataconverter import DataConverter
from graphnet.data.extractors.icecube import I3Extractor
from graphnet.data.extractors.icecube.utilities.frames import (
    frame_is_montecarlo,
    frame_is_noise,
    get_om_keys_and_pulseseries,
)
from graphnet.data.extractors.icecube.utilities.i3_filters import (
    I3Filter,
    NullSplitI3Filter,
)
from graphnet.data.readers import I3Reader
from graphnet.data.writers import ParquetWriter
from graphnet.utilities.filesys import find_i3_files

# ============================================================
# Reader
# ============================================================

class PONE_Reader(I3Reader):
    """A class for reading .i3 files for P_ONE (LeptonInjector Simulation).

    Note that this class relies on IceCube-specific software, and
    therefore must be run in a software environment that contains
    IceTray.
    """

    def __init__(
        self,
        gcd_rescue: str,
        i3_filters: Optional[Union[I3Filter, List[I3Filter]]] = None,
        pulsemap: str = "EventPulseSeries",
        skip_empty_pulses: bool = True,
    ):
        """Initialize `PONE_Reader`.

        Args:
            gcd_rescue: Path to a GCD file that will be used if no GCD file is
                        found in subfolder. `PONE_Reader` will recursively search
                        the input directory for I3-GCD file pairs. By convention,
                        a folder containing i3 files will have an
                        accompanying GCD file. However, in some cases, this
                        convention is broken. In cases where a folder contains
                        i3 files but no GCD file, the `gcd_rescue` is used
                        instead.
            i3_filters: Instances of `I3Filter` to filter PFrames. Defaults to
                        `NullSplitI3Filter`.
        """
        super().__init__(gcd_rescue=gcd_rescue, i3_filters=i3_filters)
        self._pulsemap = pulsemap
        self._skip_empty_pulses = skip_empty_pulses

    @property
    def extractor_names(self) -> list[str]:
        return self.extracor_names

    def __call__(
        self, file_path: I3FileSet
    ) -> List[OrderedDict]:  # noqa: E501  # type: ignore
        """Extract data from single I3 file.

        Args:
            fileset: Path to I3 file and corresponding GCD file.

        Returns:
            Extracted data.
        """
        # Set I3-GCD file pair in extractor
        for extractor in self._extractors:
            assert isinstance(extractor, I3Extractor)
            extractor.set_gcd(
                i3_file=file_path.i3_file, gcd_file=file_path.gcd_file
            )

        # Open I3 file
        i3_file_io = dataio.I3File(file_path.i3_file, "r")
        data = list()
        while i3_file_io.more():
            try:
                frame = i3_file_io.pop_daq()
            except Exception:
                continue

            # check if frame should be skipped
            if self._skip_frame(frame):
                continue

            # Try to extract data from I3Frame
            results = [extractor(frame) for extractor in self._extractors]
            data_dict = OrderedDict(zip(self.extractor_names, results))
            data.append(data_dict)

        return data

    def _skip_frame(self, frame: "icetray.I3Frame") -> bool:
        """Skip frame if base filters fail OR if pulsemap missing/empty."""
        # 1) base class filters (NullSplitI3Filter etc.)
        if super()._skip_frame(frame):
            return True

        # 2) pulsemap must exist
        if self._pulsemap not in frame:
            return True

        # 3) optionally skip empty pulse maps
        if self._skip_empty_pulses:
            pmap = frame[self._pulsemap]
            # For I3RecoPulseSeriesMap etc., len(pmap) usually gives number of OMKeys
            try:
                if len(pmap) == 0:
                    return True
            except TypeError:
                # fallback if len() not supported
                try:
                    if len(list(pmap.keys())) == 0:
                        return True
                except Exception:
                    # If we can't determine emptiness reliably, do not skip
                    pass

        return False


# ============================================================
# Feature extractor
# ============================================================

class I3FeatureExtractorPONE(I3Extractor):
    """Class for extracting reconstructed features for P-ONE events created with LeptonInjector."""

    def __init__(
        self,
        pulsemap: str,
        name: str = "feature",
        exclude: list = [None],
    ):
        # keep same behavior / naming as your current class
        self._pulsemap = pulsemap

        # Base class constructor
        super().__init__(name=name, exclude=exclude)
        self._extractor_name = name

        self._detector_status: Optional["icetray.I3Frame.DetectorStatus"] = None

        self._PMT_ANGLES = np.array(
            [
                [57.5, 270.],   # PMT 1
                [57.5, 0.],     # PMT 2
                [57.5, 90.],    # PMT 3
                [57.5, 180.],   # PMT 4
                [25., 225.],    # PMT 5
                [25., 315.],    # PMT 6
                [25., 45.],     # PMT 7
                [25., 135.],    # PMT 8
                [-57.5, 270.],  # PMT 9
                [-57.5, 180.],  # PMT 10
                [-57.5, 90.],   # PMT 11
                [-57.5, 0.],    # PMT 12
                [-25., 315.],   # PMT 13
                [-25., 225.],   # PMT 14
                [-25., 135.],   # PMT 15
                [-25., 45.],    # PMT 16
            ]
        )

        self.MODULE_RADIUS_M = 0.2159

        self._pmt_x_coordinates_wrt_om_rotated = np.multiply(
            np.sin(np.deg2rad(90.0 - self._PMT_ANGLES[:, 0])),
            np.cos(np.deg2rad(self._PMT_ANGLES[:, 1])),
        )
        self._pmt_y_coordinates_wrt_om_rotated = np.multiply(
            np.sin(np.deg2rad(90.0 - self._PMT_ANGLES[:, 0])),
            np.sin(np.deg2rad(self._PMT_ANGLES[:, 1])),
        )
        self._pmt_z_coordinates_wrt_om_rotated = np.cos(
            np.deg2rad(90.0 - self._PMT_ANGLES[:, 0])
        )

        self._PMT_MATRIX_rotated = np.array(
            [
                self._pmt_x_coordinates_wrt_om_rotated,
                self._pmt_y_coordinates_wrt_om_rotated,
                self._pmt_z_coordinates_wrt_om_rotated,
            ]
        ).T

        self._PMT_COORDINATES_rotated = (
            self._PMT_MATRIX_rotated * self.MODULE_RADIUS_M
        )

        self._minus_90_degree_rotation_around_x_axis = np.array(
            [
                [1.000000e00, 0.000000e00, 0.000000e00],
                [0.000000e00, 0.000000e00, 1.000000e00],
                [0.000000e00, -1.000000e00, 0.000000e00],
            ],
            dtype=float,
        )

        self._PMT_COORDINATES_ORIGINAL = (
            self._PMT_COORDINATES_rotated
            @ self._minus_90_degree_rotation_around_x_axis.T
        )

    def set_gcd(self, i3_file: str, gcd_file: Optional[str] = None) -> None:
        """Extract GFrame, CFrame and DFrame from i3/gcd-file pair.

        Information from these frames will be set as member variables of
        `I3Extractor`.
        """
        super().set_gcd(i3_file=i3_file, gcd_file=gcd_file)

        if gcd_file is None:
            gcd = dataio.I3File(i3_file)
        else:
            gcd = dataio.I3File(gcd_file)

        try:
            d_frame = gcd.pop_frame(icetray.I3Frame.DetectorStatus)
        except RuntimeError as e:
            self.error(
                "No GCD file was provided "
                f"and no D-frame was found in {i3_file.split('/')[-1]}."
            )
            raise e

        self._detector_status = d_frame

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, List[Any]]:
        """Extract reconstructed features from `frame`."""
        padding_value: float = -1.0
        output: Dict[str, List[Any]] = {
            "charge": [],
            "dom_time": [],
            "width": [],
            "dom_x": [],
            "dom_y": [],
            "dom_z": [],
            "pmt_area": [],
            "rde": [],
            "is_bright_dom": [],
            "is_bad_dom": [],
            "is_saturated_dom": [],
            "is_errata_dom": [],
            "event_time": [],
            "hlc": [],
            "awtd": [],
            "string": [],
            "pmt_number": [],
            "dom_number": [],
            "dom_type": [],
            "pmt_x": [],
            "pmt_y": [],
            "pmt_z": [],
        }

        # Get OM data
        if self._pulsemap in frame:
            om_keys, data = get_om_keys_and_pulseseries(
                frame,
                self._pulsemap,
                self._calibration,
            )
        else:
            self.warning_once(f"Pulsemap {self._pulsemap} not found in frame.")
            return output

        # keep same behavior as your current class
        is_bright_dom = -1
        is_saturated_dom = -1
        is_errata_dom = -1

        bad_doms = None
        if self._detector_status is not None:
            if "BadDomsList" in self._detector_status:
                bad_doms = self._detector_status["BadDomsList"]

        event_time = frame["I3EventHeader"].start_time.mod_julian_day_double

        for om_key in om_keys:
            x = self._gcd_dict[om_key].position.x
            y = self._gcd_dict[om_key].position.y
            z = self._gcd_dict[om_key].position.z
            area = self._gcd_dict[om_key].area
            rde = self._get_relative_dom_efficiency(
                frame, om_key, padding_value
            )

            string = om_key[0]
            dom_number = om_key[1]
            pmt_number = om_key[2]
            dom_type = self._gcd_dict[om_key].omtype

            pmt_x = pmt_y = pmt_z = padding_value
            if pmt_number is not None:
                idx = int(pmt_number) - 1
                if 0 <= idx < len(self._PMT_COORDINATES_ORIGINAL):
                    rel = self._PMT_COORDINATES_ORIGINAL[idx]
                    pmt_x = x + float(rel[0])
                    pmt_y = y + float(rel[1])
                    pmt_z = z + float(rel[2])

            if bad_doms:
                is_bad_dom = 1 if om_key in bad_doms else 0
            else:
                is_bad_dom = int(padding_value)

            pulses = data[om_key]
            for pulse in pulses:
                output["charge"].append(
                    getattr(pulse, "charge", padding_value)
                )
                output["dom_time"].append(
                    getattr(pulse, "time", padding_value)
                )
                output["width"].append(getattr(pulse, "width", padding_value))
                output["pmt_area"].append(area)
                output["rde"].append(rde)
                output["dom_x"].append(x)
                output["dom_y"].append(y)
                output["dom_z"].append(z)
                output["pmt_x"].append(pmt_x)
                output["pmt_y"].append(pmt_y)
                output["pmt_z"].append(pmt_z)

                output["string"].append(string)
                output["pmt_number"].append(pmt_number)
                output["dom_number"].append(dom_number)
                output["dom_type"].append(dom_type)

                output["is_bad_dom"].append(is_bad_dom)
                output["event_time"].append(event_time)
                output["is_bright_dom"].append(is_bright_dom)
                output["is_errata_dom"].append(is_errata_dom)
                output["is_saturated_dom"].append(is_saturated_dom)

                flags = getattr(pulse, "flags", padding_value)
                if flags == padding_value:
                    output["hlc"].append(padding_value)
                    output["awtd"].append(padding_value)
                else:
                    output["hlc"].append((pulse.flags >> 0) & 0x1)
                    output["awtd"].append(self._parse_awtd_flag(pulse))

        return output

    def _get_relative_dom_efficiency(
        self,
        frame: "icetray.I3Frame",
        om_key: int,
        padding_value: float,
    ) -> float:
        if "I3Calibration" in frame:
            rde = frame["I3Calibration"].dom_cal[om_key].relative_dom_eff
        else:
            try:
                assert self._calibration is not None
                rde = self._calibration.dom_cal[om_key].relative_dom_eff
            except:  # noqa: E722
                rde = padding_value
        return rde

    def _parse_awtd_flag(
        self,
        pulse: Any,
        fadc_min_width_ns: float = 6.0,
    ) -> bool:
        """Parse awtd flag from pulse width."""
        return pulse.width < (fadc_min_width_ns * icetray.I3Units.ns)


# ============================================================
# Truth extractor
# ============================================================

class I3TruthExtractorPONE(I3Extractor):
    """Truth-level extractor for PONE simulations.

    Compared to I3TruthExtractor, this class:
      - Derives detector boundaries dynamically from the GCD file instead
        of relying on hardcoded IceCube coordinates.
      - Does not filter on sub_event_stream (no InIceSplit/Final requirement),
        since PONE data does not use that frame splitting convention.
      - Keeps the interaction vertex in position_xyz for all particle types.
        For muons, the final stopping position is stored separately in
        muon_final_x/y/z.
      - For primary muons (pid=±13), track_length and muon_final are computed
        from the primary particle itself.
      - For muon neutrinos (pid=±14), track_length and muon_final are computed
        from the secondary muon found among the primary's daughters.
      - For LeptonInjector simulations, energy_track, energy_cascade, and
        inelasticity are derived directly from EventProperties (finalStateY
        and totalEnergy) rather than MCTree daughter shapes. This is correct
        for all neutrino flavours (νμ, νe, ντ) and both CC and NC interactions.
    """

    def __init__(
        self,
        name: str = "truth",
        mctree: Optional[str] = "I3MCTree",
        extend_boundary: Optional[float] = 0.0,
        string_proximity_threshold: float = 100.0,
        exclude: list = [None],
    ):
        """Construct I3TruthExtractorPONE.

        Args:
            name: Name of the I3Extractor instance.
            mctree: Name of the MC tree to use for truth values.
            extend_boundary: Distance in metres to extend the convex hull of
                the detector when defining starting events. Defaults to 0.
            string_proximity_threshold: Maximum XY distance in metres from a
                point to the nearest string for it to be considered
                "near a string". Used for is_starting_near_string and
                *_stopped_near_string columns. Defaults to 100.
            exclude: List of keys to exclude from the extracted data.
        """
        super().__init__(name, exclude=exclude)

        # Borders cannot be computed here because the GCD file has not been
        # loaded yet. They will be set in set_gcd() once sensor positions
        # are available.
        self._borders = None
        self._extend_boundary = extend_boundary
        self._string_proximity_threshold = string_proximity_threshold
        self._mctree = mctree

    def set_gcd(self, i3_file: str, gcd_file: Optional[str] = None) -> None:
        """Load GCD file and derive detector geometry.

        Calls the parent implementation to populate self._gcd_dict, then
        computes detector boundaries and the Delaunay triangulation used for
        vertex containment checks — all from actual sensor positions rather
        than hardcoded coordinates.

        Args:
            i3_file: Path to the i3 file being converted.
            gcd_file: Path to the GCD file. If None, the method will look for
                G and C frames inside i3_file.
        """
        super().set_gcd(i3_file=i3_file, gcd_file=gcd_file)

        # Collect 3D sensor positions from the GCD
        coordinates = np.array([
            [g.position.x, g.position.y, g.position.z]
            for g in self._gcd_dict.values()
        ])

        # Optionally expand the boundary outward from the centroid.
        # This is useful when you want to include events that interact just
        # outside the instrumented volume.
        if self._extend_boundary != 0.0:
            center = np.mean(coordinates, axis=0)
            d = coordinates - center
            norms = np.linalg.norm(d, axis=1, keepdims=True)
            dn = d / norms
            coordinates = coordinates + dn * self._extend_boundary

        # Build the 3D convex hull and Delaunay triangulation used by
        # _contained_vertex() to test whether an interaction vertex lies
        # inside the detector.
        hull = ConvexHull(coordinates)
        self.hull = hull
        self.delaunay = Delaunay(coordinates[hull.vertices])

        # Derive the 2D (XY) boundary polygon and Z range used by
        # _muon_stopped() to test whether a muon's final position lies
        # within the fiducial volume.
        xy = coordinates[:, :2]
        hull_2d = ConvexHull(xy)
        border_xy = xy[hull_2d.vertices]
        border_z = np.array([coordinates[:, 2].min(), coordinates[:, 2].max()])
        self._borders = [border_xy, border_z]

        # Store unique string XY positions for proximity-based containment
        # checks. Strings are vertical so we only need XY; we deduplicate by
        # rounding to the nearest metre to collapse all sensors on the same
        # string into a single point.
        xy_rounded = np.round(xy, 0)
        self._string_xy = np.unique(xy_rounded, axis=0)

    def __call__(
        self, frame: "icetray.I3Frame", padding_value: Any = -1
    ) -> Dict[str, Any]:
        """Extract truth-level information from a physics frame."""
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
            # Muon produced at the νμ CC interaction vertex (pid=±14)
            "numu_muon_track_length": padding_value,
            "numu_muon_stopped_convex_hull": padding_value,
            "numu_muon_final_x": padding_value,
            "numu_muon_final_y": padding_value,
            "numu_muon_final_z": padding_value,
            # Primary atmospheric / MuonGun muon (pid=±13)
            "atmo_muon_track_length": padding_value,
            "atmo_muon_stopped_convex_hull": padding_value,
            "atmo_muon_final_x": padding_value,
            "atmo_muon_final_y": padding_value,
            "atmo_muon_final_z": padding_value,
            "energy_track": padding_value,
            "energy_cascade": padding_value,
            "inelasticity": padding_value,
            "is_starting_convex_hull": padding_value,
            # Proximity-based containment: True if the point is within
            # string_proximity_threshold metres (XY) of the nearest string
            # AND within the instrumented Z range derived from the GCD.
            "is_starting_near_string": padding_value,
            "numu_muon_stopped_near_string": padding_value,
            "atmo_muon_stopped_near_string": padding_value,
            # Tau decay muon from ντ CC interaction (pid=±16, τ → μ decay mode)
            "tau_decay_length": padding_value,
            "tau_decay_muon_track_length": padding_value,
            "tau_decay_muon_stopped_convex_hull": padding_value,
            "tau_decay_muon_stopped_near_string": padding_value,
            "tau_decay_muon_final_x": padding_value,
            "tau_decay_muon_final_y": padding_value,
            "tau_decay_muon_final_z": padding_value,
        }

        # Note: the InIceSplit / Final sub_event_stream filter present in
        # I3TruthExtractor is intentionally omitted here. PONE data does not
        # use that frame splitting convention.

        if is_mc and (not is_noise):
            (
                MCTree,
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
                ) = self._get_primary_track_energy_and_inelasticity(
                    frame, sim_type
                )
            except RuntimeError:
                # Track energy calculation fails for some northern tracks where
                # the Hadrons particle has no mass implemented.
                energy_track, energy_cascade, inelasticity = (
                    padding_value,
                    padding_value,
                    padding_value,
                )

            output.update(
                {
                    "energy": MCTree.energy,
                    "position_x": MCTree.pos.x,
                    "position_y": MCTree.pos.y,
                    "position_z": MCTree.pos.z,
                    "azimuth": MCTree.dir.azimuth,
                    "zenith": MCTree.dir.zenith,
                    "pid": MCTree.pdg_encoding,
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

            # Determine which particle to use for muon track calculations:
            # - Primary muon (pid=±13)  → atmospheric / MuonGun muon,
            #   written into atmo_muon_* columns.
            # - Muon neutrino (pid=±14) → find the secondary muon among the
            #   primary's daughters (CC interaction), written into numu_muon_*
            #   columns.
            # - All other flavours (νe, ντ) → both column groups stay at -1.
            pid = output["pid"]
            if abs(pid) == 13:
                # Primary is an atmospheric / MuonGun muon.
                muon_truth = {
                    "position_x": MCTree.pos.x,
                    "position_y": MCTree.pos.y,
                    "position_z": MCTree.pos.z,
                    "azimuth": MCTree.dir.azimuth,
                    "zenith": MCTree.dir.zenith,
                    "track_length": MCTree.length,
                }
                muon_final = self._muon_stopped(muon_truth, self._borders)
                output.update(
                    {
                        "atmo_muon_track_length": MCTree.length,
                        "atmo_muon_stopped_convex_hull": muon_final["stopped"],
                        "atmo_muon_final_x": muon_final["x"],
                        "atmo_muon_final_y": muon_final["y"],
                        "atmo_muon_final_z": muon_final["z"],
                    }
                )

            elif abs(pid) == 14:
                # Primary is a muon neutrino: find the secondary muon among
                # the primary's daughters (produced at the CC vertex).
                mc_tree = frame[self._mctree]
                daughters = mc_tree.get_daughters(mc_tree.primaries[0])
                numu_muon = next(
                    (d for d in daughters if abs(d.pdg_encoding) == 13), None
                )
                if numu_muon is not None:
                    muon_truth = {
                        "position_x": numu_muon.pos.x,
                        "position_y": numu_muon.pos.y,
                        "position_z": numu_muon.pos.z,
                        "azimuth": numu_muon.dir.azimuth,
                        "zenith": numu_muon.dir.zenith,
                        "track_length": numu_muon.length,
                    }
                    muon_final = self._muon_stopped(muon_truth, self._borders)
                    output.update(
                        {
                            "numu_muon_track_length": numu_muon.length,
                            "numu_muon_stopped_convex_hull": muon_final["stopped"],
                            "numu_muon_final_x": muon_final["x"],
                            "numu_muon_final_y": muon_final["y"],
                            "numu_muon_final_z": muon_final["z"],
                        }
                    )

            elif abs(pid) == 16:
                # Primary is a tau neutrino: find the tau among the primary's
                # daughters, then look for a muon among the tau's daughters
                # (tau → μ + νμ + ντ decay mode, ~17% branching ratio).
                mc_tree = frame[self._mctree]
                primary = mc_tree.primaries[0]
                tau_particle = next(
                    (d for d in mc_tree.get_daughters(primary)
                     if abs(d.pdg_encoding) == 15),
                    None,
                )
                if tau_particle is not None:
                    # tau_decay_length: path length of the tau lepton in metres,
                    # i.e. the distance from the neutrino interaction vertex to
                    # the tau decay point. Meaningful at PeV energies; near zero
                    # at hundreds of GeV. Stored as padding_value when the length
                    # field is NaN or zero (tau decayed at rest).
                    tau_len = tau_particle.length
                    output["tau_decay_length"] = (
                        float(tau_len)
                        if (tau_len == tau_len and tau_len > 0)
                        else padding_value
                    )

                    tau_muon = next(
                        (d for d in mc_tree.get_daughters(tau_particle)
                         if abs(d.pdg_encoding) == 13),
                        None,
                    )
                    if tau_muon is not None:
                        muon_truth = {
                            "position_x": tau_muon.pos.x,
                            "position_y": tau_muon.pos.y,
                            "position_z": tau_muon.pos.z,
                            "azimuth": tau_muon.dir.azimuth,
                            "zenith": tau_muon.dir.zenith,
                            "track_length": tau_muon.length,
                        }
                        muon_final = self._muon_stopped(muon_truth, self._borders)
                        output.update(
                            {
                                "tau_decay_muon_track_length": tau_muon.length,
                                "tau_decay_muon_stopped_convex_hull": muon_final["stopped"],
                                "tau_decay_muon_final_x": muon_final["x"],
                                "tau_decay_muon_final_y": muon_final["y"],
                                "tau_decay_muon_final_z": muon_final["z"],
                            }
                        )
                    else:
                        # τ → hadrons (~65%) or τ → e (~18%): no secondary muon.
                        # The tau daughters are pions or an electron, not a single
                        # Hadrons particle, so _extract_dbang_decay_length returns
                        # padding_value. Override it here using tau_decay_length,
                        # which is the distance from the neutrino vertex to the
                        # tau decay point — the correct inter-cascade separation.
                        if output["tau_decay_length"] != padding_value:
                            output["dbang_decay_length"] = output["tau_decay_length"]

            # is_starting_convex_hull uses the primary interaction vertex
            # (position_xyz), which is correct for all particle types.
            output.update({"is_starting_convex_hull": self._contained_vertex(output)})

            # Proximity-based containment checks (geometry-agnostic).
            # Uses XY distance to the nearest string AND the instrumented Z
            # range, so it works correctly for any detector layout including
            # non-convex geometries.
            output.update({
                "is_starting_near_string": self._near_string(
                    output["position_x"], output["position_y"], output["position_z"]
                ),
                "numu_muon_stopped_near_string": self._near_string(
                    output["numu_muon_final_x"], output["numu_muon_final_y"],
                    output["numu_muon_final_z"]
                ) if output["numu_muon_final_x"] != -1 else padding_value,
                "atmo_muon_stopped_near_string": self._near_string(
                    output["atmo_muon_final_x"], output["atmo_muon_final_y"],
                    output["atmo_muon_final_z"]
                ) if output["atmo_muon_final_x"] != -1 else padding_value,
                "tau_decay_muon_stopped_near_string": self._near_string(
                    output["tau_decay_muon_final_x"], output["tau_decay_muon_final_y"],
                    output["tau_decay_muon_final_z"]
                ) if output["tau_decay_muon_final_x"] != -1 else padding_value,
            })

        return output

    def _extract_dbang_decay_length(
        self, frame: "icetray.I3Frame", padding_value: float = -1
    ) -> float:
        """Extract the distance between the two cascades in double-bang events.

        A double-bang topology occurs in ντ CC (τ → hadrons, ~65% branching
        ratio) and HNL events. In both cases the primary has two daughters:
        a Hadrons particle at the interaction vertex (cascade 0) and a
        secondary particle (tau or HNL) that travels a short distance before
        decaying into a second Hadrons particle (cascade 1).

        For ντ CC with τ → μ (~17%) the tau's daughters contain no Hadrons
        particle, so this method correctly returns padding_value. The muon
        information is captured separately in tau_decay_muon_* columns.

        Args:
            frame: Physics frame containing MC record.
            padding_value: Value returned when the decay length cannot be
                determined or the event is not a double-bang topology.

        Returns:
            Distance in metres between the two hadronic cascades, or
            padding_value when the double-bang topology is not present.
        """
        mctree = frame[self._mctree]
        try:
            primary = mctree.primaries[0]
            daughters = mctree.get_daughters(primary)

            # Require exactly two primary daughters: Hadrons (cascade 0) and
            # a secondary particle (tau or HNL).
            if len(daughters) != 2:
                return padding_value

            casc_0 = next(
                (d for d in daughters
                 if d.type == dataclasses.I3Particle.Hadrons),
                None,
            )
            secondary = next(
                (d for d in daughters
                 if d.type != dataclasses.I3Particle.Hadrons),
                None,
            )
            if casc_0 is None or secondary is None:
                return padding_value

            # Among the secondary's decay products, look for a Hadrons
            # particle that marks the second cascade (tau → hadrons or
            # HNL → hadrons). If none is found (e.g. tau → μ mode) return
            # padding_value.
            sec_daughters = mctree.get_daughters(secondary)
            casc_1 = next(
                (d for d in sec_daughters
                 if d.type == dataclasses.I3Particle.Hadrons),
                None,
            )
            if casc_1 is None:
                return padding_value

            return (
                phys_services.I3Calculator.distance(casc_0, casc_1)
                / icetray.I3Units.m
            )
        except:  # noqa: E722
            return padding_value

    def _muon_stopped(
        self,
        truth: Dict[str, Any],
        borders: List[np.ndarray],
        shrink_horizontally: float = 100.0,
        shrink_vertically: float = 100.0,
    ) -> Dict[str, Any]:
        """Compute the muon's final position and whether it stopped inside
        the fiducial volume.

        The final position is calculated by propagating the muon from its
        starting position along its direction vector by track_length metres.

        Args:
            truth: Dictionary containing position_x/y/z, azimuth, zenith,
                and track_length of the muon particle.
            borders: [border_xy, border_z] where border_xy is an (N, 2) array
                of polygon vertices in the XY plane and border_z is [z_min, z_max].
            shrink_horizontally: Inward shrink of the XY polygon in metres.
                Defaults to 100.
            shrink_vertically: Inward shrink in the Z direction in metres.
                Defaults to 100.

        Returns:
            Dictionary with keys x, y, z (final position) and stopped (bool).
        """
        border = mpath.Path(borders[0])

        start_pos = np.array(
            [truth["position_x"], truth["position_y"], truth["position_z"]]
        )

        # Propagate along the direction vector. The -1 factor is needed because
        # zenith/azimuth in IceCube convention point toward the particle origin,
        # not its direction of travel.
        travel_vec = -1 * np.array(
            [
                truth["track_length"]
                * np.cos(truth["azimuth"])
                * np.sin(truth["zenith"]),
                truth["track_length"]
                * np.sin(truth["azimuth"])
                * np.sin(truth["zenith"]),
                truth["track_length"] * np.cos(truth["zenith"]),
            ]
        )

        end_pos = start_pos + travel_vec

        stopped_xy = border.contains_point(
            (end_pos[0], end_pos[1]), radius=-shrink_horizontally
        )
        stopped_z = (end_pos[2] > borders[1][0] + shrink_vertically) * (
            end_pos[2] < borders[1][1] - shrink_vertically
        )

        return {
            "x": end_pos[0],
            "y": end_pos[1],
            "z": end_pos[2],
            "stopped": (stopped_xy * stopped_z),
        }

    def _get_primary_particle_interaction_type_and_elasticity(
        self,
        frame: "icetray.I3Frame",
        sim_type: str,
        padding_value: float = -1.0,
    ) -> Tuple[Any, int, float]:
        """Return the primary MC particle, interaction type, and elasticity.

        Args:
            frame: Physics frame containing MC record.
            sim_type: Simulation type string (e.g. 'LeptonInjector', 'NuGen').
            padding_value: Fallback value used when a quantity cannot be read.

        Returns:
            Tuple of (MCTree, interaction_type, elasticity) where
            interaction_type is 1 (CC), 2 (NC), or padding_value.
        """
        if sim_type != "noise":
            MCTree = frame[self._mctree][0]
            if MCTree.energy != MCTree.energy:
                MCTree = frame[self._mctree][1]
        else:
            MCTree = None

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

        return MCTree, interaction_type, elasticity

    def _get_primary_track_energy_and_inelasticity(
        self,
        frame: "icetray.I3Frame",
        sim_type: str,
    ) -> Tuple[float, float, float]:
        """Compute track energy, cascade energy, and inelasticity.

        For LeptonInjector simulations, all three quantities are derived
        directly from EventProperties (finalStateY and totalEnergy). This is
        correct for all neutrino flavours (νμ, νe, ντ) and for both CC and NC
        interactions, since EventProperties always records the true lepton /
        hadron energy split at the interaction vertex, regardless of what
        happens during propagation.

        The MCTree-based fallback (StartingTrack / Dark shape filter) is kept
        for NuGen and other sim types where EventProperties is unavailable. It
        is only reliable for νμ CC events; for νe and ντ the track shape may
        be absent, causing energy_track to be underestimated.

        Args:
            frame: Physics frame containing MC record.
            sim_type: Simulation type string (e.g. 'LeptonInjector', 'NuGen').

        Returns:
            Tuple of (energy_track, energy_cascade, inelasticity).
        """
        if sim_type == "LeptonInjector":
            ep = frame["EventProperties"]
            y = ep.finalStateY
            e_total = ep.totalEnergy

            energy_track = e_total * (1.0 - y)
            energy_cascade = e_total * y
            inelasticity = y

            return energy_track, energy_cascade, inelasticity

        # ── fallback: NuGen / other sim types ────────────────────────────
        mc_tree = frame[self._mctree]
        primary = mc_tree.primaries[0]
        daughters = mc_tree.get_daughters(primary)

        tracks = [
            d for d in daughters
            if str(d.shape) in ("StartingTrack", "Dark")
        ]

        energy_total = primary.total_energy
        energy_track = sum(t.total_energy for t in tracks)
        energy_cascade = energy_total - energy_track
        inelasticity = 1.0 - energy_track / energy_total

        return energy_track, energy_cascade, inelasticity

    def _find_data_type(
        self, mc: bool, input_file: str, frame: "icetray.I3Frame"
    ) -> str:
        """Determine the simulation or data type from the file path and frame.

        Args:
            mc: Whether the input file is Monte Carlo simulation.
            input_file: Path to the i3 file.
            frame: Physics frame, used as fallback when path is uninformative.

        Returns:
            A string identifying the data type, e.g. 'LeptonInjector', 'NuGen',
            'muongun', 'corsika', 'genie', 'noise', or 'data'.
        """
        if not mc:
            sim_type = "data"
        elif frame.Has("EventProperties") or frame.Has(
            "LeptonInjectorProperties"
        ):
            sim_type = "LeptonInjector"
        else:
            raise NotImplementedError("Could not determine data type.")
        return sim_type

    def _near_string(self, x: float, y: float, z: float) -> bool:
        """Check whether a point (x, y, z) is within the instrumented volume.

        A point is considered inside if:
          1. Its XY distance to the nearest string is within
             self._string_proximity_threshold metres, AND
          2. Its Z coordinate lies within the instrumented depth range
             [z_min, z_max] derived from the GCD sensor positions.

        Unlike the convex-hull method, this works correctly for non-convex and
        multi-cluster detector geometries because it does not assume that the
        space between strings is instrumented.

        Args:
            x: X coordinate of the point in metres.
            y: Y coordinate of the point in metres.
            z: Z coordinate of the point in metres.

        Returns:
            True if the point is within the proximity threshold of the nearest
            string AND within the instrumented Z range, False otherwise.
        """
        dists = np.sqrt(
            (self._string_xy[:, 0] - x) ** 2
            + (self._string_xy[:, 1] - y) ** 2
        )
        near_xy = dists.min() <= self._string_proximity_threshold
        in_z = (self._borders[1][0] <= z <= self._borders[1][1])
        return bool(near_xy and in_z)

    def _contained_vertex(self, truth: Dict[str, Any]) -> bool:
        """Check whether an interaction vertex lies inside the detector volume.

        Uses the Delaunay triangulation built from GCD sensor positions in
        set_gcd().

        Args:
            truth: Dictionary of already extracted truth-level information.

        Returns:
            True if the vertex is inside the detector, False otherwise.
        """
        vertex = np.array(
            [truth["position_x"], truth["position_y"], truth["position_z"]]
        )
        return self.delaunay.find_simplex(vertex) >= 0


# ============================================================
# I3 filters
# ============================================================

class NonEmptyPulseSeriesI3Filter(I3Filter):
    """Drop frame if given PulseSeriesMap/Mask is empty (has 0 pulses)."""

    def __init__(self, pulsemap_name: str = "EventPulseSeries_nonoise"):
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._pulsemap_name = pulsemap_name

    def _keep_frame(self, frame: "icetray.I3Frame") -> bool:
        if not frame.Has(self._pulsemap_name):
            return False

        pm = frame[self._pulsemap_name]

        # Handle MapMask vs Map
        try:
            if hasattr(pm, "apply"):
                pm = pm.apply(frame)
        except Exception:
            # If apply fails for any reason, do not keep the frame
            return False

        # Count total pulses across all OMKeys/PMTs
        try:
            total = 0
            for _, series in pm.items():
                total += len(series)
                if total > 0:
                    return True
            return False
        except Exception:
            # Fallback: if it behaves like a container
            try:
                return len(pm) > 0
            except Exception:
                return False


# ============================================================
# Paths / config
# ============================================================



geometry = os.environ["PONE_GEOMETRY"]
flavor = os.environ["PONE_FLAVOR"]

INPUT_GLOB = f"/scratch/kbas/{geometry}/{flavor}_PMT_Response/*.i3.gz"
INPUT_ROOT = f"/scratch/kbas/{geometry}/{flavor}_PMT_Response/"
OUTDIR = f"/scratch/kbas/FinalParquetDatasets/{geometry}_{flavor.lower()}_nonoise"
GCD_RESCUE = f"/scratch/kbas/{geometry}/GCD_{geometry.replace('_string', 'strings')}.i3.gz"


# ============================================================
# Helper functions
# ============================================================

def batch_id_from_i3(path):
    m = re.search(r"muon_batch_(\d+)\.i3\.gz$", os.path.basename(path))
    return int(m.group(1)) if m else None


def batch_ids_in_outdir(outdir):
    # Scan anything containing "muon_batch_<id>" under outdir
    candidates = glob.glob(os.path.join(outdir, "**", "*"), recursive=True)
    ids = set()
    for p in candidates:
        m = re.search(r"muon_batch_(\d+)", os.path.basename(p))
        if m:
            ids.add(int(m.group(1)))
    return ids


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hardlink preferred
    except Exception:
        try:
            os.symlink(src, dst)  # fallback symlink
        except Exception:
            shutil.copy2(src, dst)  # fallback copy


def reindex_split(old_split_dir: str, new_split_dir: str):
    old = Path(old_split_dir)
    new = Path(new_split_dir)
    (new / "truth").mkdir(parents=True, exist_ok=True)
    (new / "features").mkdir(parents=True, exist_ok=True)

    rx = re.compile(r"_(\d+)\.parquet$")
    ids = sorted(
        int(rx.search(p.name).group(1))
        for p in (old / "truth").glob("truth_*.parquet")
    )

    for new_id, old_id in enumerate(ids):
        for table in ["truth", "features"]:
            src = old / table / f"{table}_{old_id}.parquet"
            dst = new / table / f"{table}_{new_id}.parquet"
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)


# ============================================================
# Discover inputs / already-processed batches
# ============================================================

all_files = sorted(glob.glob(INPUT_GLOB))
done_ids = batch_ids_in_outdir(OUTDIR)


# ============================================================
# Reader, extractors, writer, converter
# ============================================================

reader = PONE_Reader(
    gcd_rescue=GCD_RESCUE,
    i3_filters=[
        NullSplitI3Filter(),
        NonEmptyPulseSeriesI3Filter("EventPulseSeries_nonoise"),
    ],
)

extractors = [
    I3FeatureExtractorPONE(
        pulsemap="EventPulseSeries_nonoise",
        name="features",
        exclude=[
            "pmt_area",
            "rde",
            "width",
            "event_time",
            "is_bright_dom",
            "is_saturated_dom",
            "is_errata_dom",
            "is_bad_dom",
            "hlc",
            "awtd",
            "dom_type",
        ],
    ),
    I3TruthExtractorPONE(
        mctree="I3MCTree_postprop",
        name="truth",
        exclude=[],
    ),
]

writer = ParquetWriter(truth_table="truth", index_column="event_no")

converter = DataConverter(
    file_reader=reader,
    save_method=writer,
    extractors=extractors,
    outdir=OUTDIR,
    num_workers=16,
    index_column="event_no",
)


# ============================================================
# Convert I3 -> Parquet
# ============================================================

converter(input_dir=INPUT_ROOT)

print("DONE:", OUTDIR)


# ============================================================
# Merge batches
# ============================================================

MERGED_DIR = os.path.join(OUTDIR, "merged")

writer.merge_files(
    files=[],
    output_dir=MERGED_DIR,
    events_per_batch=1024,
    num_workers=16,
)


# ============================================================
# Rename merged -> merged_raw
# ============================================================

MERGED_DIR = Path(MERGED_DIR)
MERGED_RAW = MERGED_DIR.parent / "merged_raw"

# Move merged -> merged_raw (if not already done)
if not MERGED_RAW.exists():
    os.rename(MERGED_DIR, MERGED_RAW)
    print("moved:", MERGED_DIR, "->", MERGED_RAW)
else:
    print("merged_raw already exists:", MERGED_RAW)


# ============================================================
# Build train/val/test split over merged batches
# ============================================================

truth_dir = MERGED_RAW / "truth"
feat_dir = MERGED_RAW / "features"

pat = re.compile(r"^truth_(\d+)\.parquet$")
batch_ids = sorted(
    int(pat.match(p.name).group(1))
    for p in truth_dir.glob("truth_*.parquet")
    if pat.match(p.name)
)

last_batch = max(batch_ids)
main_batches = [b for b in batch_ids if b != last_batch]

seed = 42
rng = random.Random(seed)
rng.shuffle(main_batches)

n = len(main_batches)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

splits = {
    "train": main_batches[:n_train],
    "val": main_batches[n_train: n_train + n_val],
    "test": main_batches[n_train + n_val:] + [last_batch],
}


# ============================================================
# Materialize split directories (hardlink -> symlink -> copy)
# ============================================================

NEW_MERGED = MERGED_RAW.parent / "merged"
tables = ["truth", "features"]

for split_name, ids in splits.items():
    for table in tables:
        for bid in ids:
            src = MERGED_RAW / table / f"{table}_{bid}.parquet"
            dst = NEW_MERGED / split_name / table / src.name
            if src.exists():
                link_or_copy(src, dst)

manifest = {
    "seed": seed,
    "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
    "last_batch_forced_to_test": int(last_batch),
    "counts_in_batches": {k: len(v) for k, v in splits.items()},
    "splits": splits,
}

with open(NEW_MERGED / "split_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("done:", NEW_MERGED)
print({k: len(v) for k, v in splits.items()})


# ============================================================
# Reindex splits (create *_reindexed directories using symlinks)
# ============================================================

for split_name in ["train", "val", "test"]:
    reindex_split(
        str(NEW_MERGED / split_name),
        str(NEW_MERGED / f"{split_name}_reindexed"),
    )