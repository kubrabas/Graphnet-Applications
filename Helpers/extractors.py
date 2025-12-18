### THESE ARE OLD VERSIONS





# check these from scratch
# orijinal graphnet mantigina uyuyor mu bunlar


from typing import List, Union, OrderedDict, Any, Dict, Optional
from abc import ABC, abstractmethod
import glob
import os

import pandas as pd




from graphnet.utilities.decorators import final
from graphnet.utilities.logging import Logger
from graphnet.data.dataclasses import I3FileSet
from graphnet.data.extractors.icecube import I3Extractor
from graphnet.utilities.imports import has_icecube_package
from graphnet.data.extractors.icecube.utilities.i3_filters import (
    I3Filter,
    NullSplitI3Filter,
)
from graphnet.utilities.filesys import find_i3_files

if has_icecube_package():
    from icecube import icetray, dataio  # type: ignore[import]




class PONE_TruthExtractor(I3Extractor):
    """
    Truth / label extractor for P-ONE LeptonInjector simulations.

    For each Physics frame it returns:
      - run_id, sub_run_id, event_id, sub_event_id
      - zenith, azimuth
      - x, y, z
      - initialType, finalType1, finalType2
      - totalEnergy
    """

    def __init__(
        self,
        name: str = "pone_truth",
        exclude: Optional[List[Any]] = None,
    ):
        if exclude is None:
            exclude = [None]
        super().__init__(extractor_name=name, exclude=exclude)

    def __call__(self, frame) -> Dict[str, Any]:
        """Extract truth information from a single Physics frame."""
        if "I3EventHeader" not in frame or "EventProperties" not in frame:
            return {}

        header = frame["I3EventHeader"]
        ep = frame["EventProperties"]

        row = {
            "run_id": int(header.run_id),
            "sub_run_id": int(header.sub_run_id),
            "event_id": int(header.event_id),
            "sub_event_id": int(header.sub_event_id),
            "zenith": float(ep.zenith),
            "azimuth": float(ep.azimuth),
            "x": float(ep.x),
            "y": float(ep.y),
            "z": float(ep.z),
            "initialType": int(ep.initialType),
            "finalType1": int(ep.finalType1),
            "finalType2": int(ep.finalType2),
            "totalEnergy": float(ep.totalEnergy),
        }
        return row


class PONE_FeatureExtractor(I3Extractor):
    """
    Pulse-level feature extractor for P-ONE.

    For each Physics frame, it produces one row per pulse:
      - run_id, sub_run_id, event_id, sub_event_id
      - string_id, om_id, pmt_id
      - time, charge
      - om_x, om_y, om_z (joined from GCD geometry via set_gcd)
    """

    COLS_BASE = [
        "run_id",
        "sub_run_id",
        "event_id",
        "sub_event_id",
        "string_id",
        "om_id",
        "pmt_id",
        "time",
        "charge",
    ]

    def __init__(
        self,
        name: str = "pone_pulses",
        pulse_series_key: str = "EventPulseSeries",
        exclude: Optional[List[Any]] = None,
    ):
        if exclude is None:
            exclude = [None]
        super().__init__(extractor_name=name, exclude=exclude)

        self._pulse_key = pulse_series_key
        self._geom_df: Optional[pd.DataFrame] = None
        self._current_gcd_file: Optional[str] = None

    # --------------------- GCD handling --------------------- #
    def set_gcd(self, i3_file: str, gcd_file: str) -> None:  # type: ignore[override]
        """
        Called by the reader once per I3/GCD file pair.

        We use this hook to load the geometry table from the GCD file.
        """
        # In case the base class defines something here:
        try:
            super().set_gcd(i3_file=i3_file, gcd_file=gcd_file)  # type: ignore[arg-type]
        except TypeError:
            # Ignore if parent signature is different
            pass

        if (self._current_gcd_file is None) or (self._current_gcd_file != gcd_file):
            self._geom_df = self._load_geometry(gcd_file)
            self._current_gcd_file = gcd_file

    def _load_geometry(self, gcd_path: str) -> pd.DataFrame:
        """
        Read GCD file and return a geometry table with:
        (string_id, om_id, pmt_id, om_x, om_y, om_z)
        """
        if not has_icecube_package():
            raise RuntimeError("IceCube software not available.")

        gcd_file = dataio.I3File(gcd_path, "r")  # type: ignore[name-defined]

        # Standard GCD layout: Geometry, Calibration, DetectorStatus
        geometry_frame = gcd_file.pop_frame()  # Geometry
        _calibration_frame = gcd_file.pop_frame()
        _status_frame = gcd_file.pop_frame()

        omgeo_map = geometry_frame["I3OMGeoMap"]

        rows: List[Dict[str, Any]] = []
        for omkey, omgeo in omgeo_map.items():
            pos = omgeo.position
            rows.append(
                {
                    "string_id": int(omkey.string),
                    "om_id": int(omkey.om),
                    # In P-ONE OMKey usually has pmt; if not, default to 0.
                    "pmt_id": int(getattr(omkey, "pmt", 0)),
                    "om_x": float(pos.x),
                    "om_y": float(pos.y),
                    "om_z": float(pos.z),
                }
            )

        return pd.DataFrame(rows)

    def _empty_df(self) -> pd.DataFrame:
        cols = list(self.COLS_BASE) + ["om_x", "om_y", "om_z"]
        return pd.DataFrame(columns=cols)

    # --------------------- main extraction --------------------- #
    def __call__(self, frame) -> pd.DataFrame:
        """
        Extract pulse-level features (and join geometry) for a single
        Physics frame.
        """
        # If header or pulse series is missing, return an empty DataFrame
        if "I3EventHeader" not in frame or self._pulse_key not in frame:
            return self._empty_df()

        header = frame["I3EventHeader"]
        pulses_map = frame[self._pulse_key]

        run_id = int(header.run_id)
        sub_run_id = int(header.sub_run_id)
        event_id = int(header.event_id)
        sub_event_id = int(header.sub_event_id)

        rows: List[Dict[str, Any]] = []

        for omkey in pulses_map:  # omkey: OMKey
            series = pulses_map[omkey]  # I3RecoPulseSeries (iterable)

            string_id = int(omkey.string)
            om_id = int(omkey.om)
            pmt_id = int(getattr(omkey, "pmt", 0))

            for pulse in series:  # one row per pulse
                rows.append(
                    {
                        "run_id": run_id,
                        "sub_run_id": sub_run_id,
                        "event_id": event_id,
                        "sub_event_id": sub_event_id,
                        "string_id": string_id,
                        "om_id": om_id,
                        "pmt_id": pmt_id,
                        "time": float(pulse.time),
                        "charge": float(pulse.charge),
                    }
                )

        if not rows:
            return self._empty_df()

        df = pd.DataFrame(rows, columns=self.COLS_BASE)

        # Join geometry if available
        if self._geom_df is not None and not self._geom_df.empty:
            df = df.merge(
                self._geom_df,
                on=["string_id", "om_id", "pmt_id"],
                how="left",
            )
        else:
            # Ensure columns exist even if geometry was not loaded
            df["om_x"] = pd.NA
            df["om_y"] = pd.NA
            df["om_z"] = pd.NA

        return df
