from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, OrderedDict 
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray, dataclasses, dataio  # pyright: reportMissingImports=false

from graphnet.data.readers import (
    GraphNeTFileReader,
    I3Reader
)

from graphnet.data.extractors.icecube.utilities.i3_filters import (
    I3Filter,
    NullSplitI3Filter,
)

from graphnet.data.extractors.icecube import I3Extractor
from graphnet.data.dataclasses import I3FileSet
from graphnet.utilities.filesys import find_i3_files


if has_icecube_package():
    from icecube import icetray, dataio  # pyright: reportMissingImports=false


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
                frame = i3_file_io.pop_physics()
            except Exception as e:
                if "I3" in str(e):
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
    
## Rasmus has a line "if "frame" in locals():" in his convert_data file, but I deleted that line. was it necessary?