import pandas as pd
from icecube import dataio, icetray


class FrameKeyToTable:
    """
    Extract a given frame key (e.g. 'EventProperties', 'I3EventHeader')
    from Physics frames in an I3 file (batch) into a pandas DataFrame.
    """

    def __init__(self, data_path: str, frame_key: str,
                 stop_type: icetray.I3Frame.Stop = icetray.I3Frame.Physics):
        """
        :param data_path: Path to the .i3 / .i3.gz file (batch).
        :param frame_key: Frame key to extract (e.g. "EventProperties").
        :param stop_type: Which frame stop to keep (default: Physics).
        """
        self.data_path = data_path
        self.frame_key = frame_key
        self.stop_type = stop_type

    def _object_to_row(self, obj) -> dict:
        """Turn an arbitrary I3 object into a flat dict."""
        row = {}
        for name in dir(obj):
            if name.startswith("_"):
                continue
            value = getattr(obj, name)
            if callable(value):
                continue
            row[name] = value
        return row

    def to_dataframe(self, max_events: int | None = None) -> pd.DataFrame:
        """
        Iterate over frames, collect the frame_key object attributes,
        and return them as a pandas DataFrame.

        :param max_events: Optional limit on number of frames to read.
        """
        rows = []
        data_file = dataio.I3File(self.data_path)

        n = 0
        while data_file.more():
            frame = data_file.pop_frame()

            # 1) only selected stop type (default: Physics)
            if frame.Stop != self.stop_type:
                continue

            # 2) must have the given key
            if self.frame_key not in frame:
                continue

            obj = frame[self.frame_key]

            # 3) object → dict
            row = self._object_to_row(obj)
            rows.append(row)

            n += 1
            if max_events is not None and n >= max_events:
                break

        return pd.DataFrame(rows)
