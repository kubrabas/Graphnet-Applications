import math
import pandas as pd
from icecube import dataio, icetray


class FrameKeyToTable:
    """
    Convert a given frame key (e.g. 'EventProperties', 'I3EventHeader',
    'EventPulseSeries', 'PMT_Response', 'PMT_Response_nonoise', 'Noise_K40', 'Noise_Dark') from 
    Physics frames in an I3 file into a
    pandas DataFrame.
    """

    SUPPORTED_KEYS = {"EventProperties", "I3EventHeader", "EventPulseSeries", 
                      "PMT_Response", "PMT_Response_nonoise", "Noise_K40", "Noise_Dark"}

    def __init__(
        self,
        data_path: str,
        frame_key: str,
        stop_type: icetray.I3Frame.Stop = icetray.I3Frame.Physics,
    ):
        """
        Parameters
        ----------
        data_path:
            Path to the .i3.gz file.
        frame_key:
            Frame key to convert (e.g. "EventProperties").
        stop_type:
            Frame stop to keep (default: Physics).
        """
        self.data_path = data_path
        self.frame_key = frame_key
        self.stop_type = stop_type

        if self.frame_key not in self.SUPPORTED_KEYS:
            raise NotImplementedError(
                f"FrameKeyToTable currently supports only "
                f"{sorted(self.SUPPORTED_KEYS)}, got '{self.frame_key}'."
            )

    # -------- dispatch: choose how to build one row from the frame object -----

    def _object_to_row(self, obj) -> dict:
        """Route the frame object to the appropriate row builder."""
        if self.frame_key in {"EventProperties", "I3EventHeader"}:
            return self._attributes_to_row(obj)
        if self.frame_key in {"EventPulseSeries", "PMT_Response", "PMT_Response_nonoise",
                             "Noise_K40", "Noise_Dark"}:
            return self._pulse_series_to_summary_row(obj)
        raise NotImplementedError(
            f"No row-conversion implemented for frame_key={self.frame_key!r}"
        )

    def _attributes_to_row(self, obj) -> dict:
        """
        Generic path: use public attributes of a simple object
        (e.g. EventProperties, I3EventHeader).
        """
        row = {}
        for name in dir(obj):
            if name.startswith("_"):
                continue
            value = getattr(obj, name)
            if callable(value):
                continue
            row[name] = value
        return row

    def _pulse_series_to_summary_row(self, pulse_series_map) -> dict:
        """
        Special path for I3RecoPulseSeriesMap-like objects
        (e.g. 'EventPulseSeries', 'PMT_Response', 'PMT_Response_nonoise', 'Noise_K40', 'Noise_Dark').
        Builds a single summary row (min/max features) for ONE event.
        """

        string_ids = []
        om_ids = []
        pmt_ids = []

        times = []
        charges = []
        widths = []
        flags = []
        total_pulses = 0

        for omkey, pulses in pulse_series_map.items():
            # OMKey: string / OM / (maybe) PMT
            if hasattr(omkey, "string"):
                string_ids.append(int(omkey.string))
            if hasattr(omkey, "om"):
                om_ids.append(int(omkey.om))
            if hasattr(omkey, "pmt"):
                pmt_ids.append(int(omkey.pmt))

            # pulses: list of I3RecoPulse
            for p in pulses:
                total_pulses += 1
                if hasattr(p, "time"):
                    times.append(float(p.time))
                if hasattr(p, "charge"):
                    charges.append(float(p.charge))
                if hasattr(p, "width"):
                    widths.append(float(p.width))
                if hasattr(p, "flags"):
                    flags.append(p.flags)

        def safe_min(xs):
            return min(xs) if xs else None

        def safe_max(xs):
            return max(xs) if xs else None

        all_width_nan = bool(widths) and all(math.isnan(w) for w in widths)
        all_flags_empty = bool(flags) and all(
            (f is None) or (f == 0) for f in flags
        )

        row = {
            # geometry-like IDs
            "min_string_id": safe_min(string_ids),
            "max_string_id": safe_max(string_ids),
            "min_om_id": safe_min(om_ids),
            "max_om_id": safe_max(om_ids),
            "min_pmt_id": safe_min(pmt_ids),
            "max_pmt_id": safe_max(pmt_ids),

            # pulse quantities
            "min_time": safe_min(times),
            "max_time": safe_max(times),
            "min_charge": safe_min(charges),
            "max_charge": safe_max(charges),

            # sanity flags
            "all_width_nan": all_width_nan,
            "all_flags_empty": all_flags_empty,

            # total number of I3RecoPulse objects in this event
            "total_pulses": total_pulses,
        }

        return row

    # ----------------------------- public API ---------------------------------

    def to_dataframe(self, max_events: int | None = None) -> pd.DataFrame:
        """
        Iterate over frames, convert the selected frame_key into one row
        per event, and return a pandas DataFrame.

        For:
          - 'EventProperties' / 'I3EventHeader' -> one column per attribute
          - 'EventPulseSeries' / 'PMT_Response' / 'PMT_Response_nonoise' / 'Noise_K40' / 'Noise_Dark' -> 
          summary columns (min/max, flags)
        """
        rows = []
        data_file = dataio.I3File(self.data_path)

        n = 0
        while data_file.more():
            frame = data_file.pop_frame()

            if frame.Stop != self.stop_type:
                continue

            if self.frame_key not in frame:
                continue

            obj = frame[self.frame_key]
            row = self._object_to_row(obj)
            rows.append(row)

            n += 1
            if max_events is not None and n >= max_events:
                break

        return pd.DataFrame(rows)
