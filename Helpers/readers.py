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



# =====================================================================
# 1) Base reader (minimal, only cares about I3Extractor)
# =====================================================================
class GraphNeTFileReader(Logger, ABC):
    """Generic base class for FileReaders in GraphNeT."""

    _accepted_file_extensions: List[str] = []
    _accepted_extractors: List[Any] = []

    @abstractmethod
    def __call__(
        self, file_path: Any
    ) -> Union[List[OrderedDict[str, pd.DataFrame]], Dict[str, pd.DataFrame]]:
        """Open and apply extractors to a single file."""
        ...

    @property
    def accepted_file_extensions(self) -> List[str]:
        return self._accepted_file_extensions

    @property
    def accepted_extractors(self) -> List[Any]:
        return self._accepted_extractors

    @property
    def extracor_names(self) -> List[str]:
        return [extractor.name for extractor in self._extractors]

    def find_files(
        self, path: Union[str, List[str]]
    ) -> Union[List[str], List[I3FileSet]]:
        """Search directory for input files recursively (simple version)."""
        if isinstance(path, str):
            path = [path]
        files: List[str] = []
        for directory in path:
            for ext in self.accepted_file_extensions:
                files.extend(glob.glob(directory + f"/*{ext}"))

        self.validate_files(files)
        return files

    @final
    def set_extractors(
        self,
        extractors: Union[List[I3Extractor], I3Extractor],
    ) -> None:
        """Attach the I3Extractor instances used by this reader."""
        if not isinstance(extractors, list):
            extractors = [extractors]
        self._validate_extractors(extractors)
        self._extractors = extractors

    @final
    def _validate_extractors(self, extractors: List[I3Extractor]) -> None:
        """Check that all extractors are instances of the accepted types."""
        for extractor in extractors:
            try:
                assert isinstance(
                    extractor, tuple(self.accepted_extractors)  # type: ignore[arg-type]
                )
            except AssertionError as e:
                self.error(
                    f"{extractor.__class__.__name__}"
                    f" is not supported by {self.__class__.__name__}"
                )
                raise e

    @final
    def validate_files(
        self, input_files: Union[List[str], List[I3FileSet]]
    ) -> None:
        """Check that the input files have accepted extensions."""
        for input_file in input_files:
            if isinstance(input_file, I3FileSet):
                self._validate_file(input_file.i3_file)
                self._validate_file(input_file.gcd_file)
            else:
                self._validate_file(input_file)

    @final
    def _validate_file(self, file: str) -> None:
        try:
            assert file.lower().endswith(tuple(self.accepted_file_extensions))
        except AssertionError:
            self.error(
                f"{self.__class__.__name__} accepts "
                f'{self.accepted_file_extensions} but {file.split("/")[-1]} '
                f"has extension {os.path.splitext(file)[1]}."
            )



# =====================================================================
# 2) P-ONE I3 reader (same logic as GraphNeT's original I3Reader)
# =====================================================================
class PONEI3Reader(GraphNeTFileReader):
    """Reader for .i3 files (IceCube / P-ONE data)."""

    def __init__(
        self,
        gcd_rescue: str,
        i3_filters: Optional[Union[I3Filter, List[I3Filter]]] = None,
        icetray_verbose: int = 0,
    ):
        assert isinstance(gcd_rescue, str)

        # Control IceTray verbosity
        if icetray_verbose == 0 and has_icecube_package():
            icetray.I3Logger.global_logger = icetray.I3NullLogger()  # type: ignore[attr-defined]

        if i3_filters is None:
            i3_filters = [NullSplitI3Filter()]

        self._accepted_file_extensions = [".bz2", ".zst", ".gz"]
        self._accepted_extractors = [I3Extractor]
        self._gcd_rescue = gcd_rescue
        self._i3filters = (
            i3_filters if isinstance(i3_filters, list) else [i3_filters]
        )

        super().__init__(name=__name__, class_name=self.__class__.__name__)

    def __call__(self, file_path: I3FileSet) -> List[OrderedDict]:
        """Extract data from a single I3 file with its GCD."""
        # Tell extractors which I3 / GCD file they are working with
        for extractor in self._extractors:
            assert isinstance(extractor, I3Extractor)
            extractor.set_gcd(
                i3_file=file_path.i3_file,
                gcd_file=file_path.gcd_file,
            )

        if not has_icecube_package():
            raise RuntimeError("IceCube software not available.")

        i3_file_io = dataio.I3File(file_path.i3_file, "r")  # type: ignore[name-defined]
        data: List[OrderedDict] = []

        while i3_file_io.more():  # type: ignore[attr-defined]
            try:
                frame = i3_file_io.pop_physics()  # type: ignore[attr-defined]
            except Exception as e:
                if "I3" in str(e):
                    continue

            if self._skip_frame(frame):
                continue

            results = [extractor(frame) for extractor in self._extractors]
            data_dict = OrderedDict(zip(self.extracor_names, results))
            data.append(data_dict)

        return data

    def find_files(self, path: Union[str, List[str]]) -> List[I3FileSet]:
        """Find I3/GCD file pairs recursively."""
        i3_files, gcd_files = find_i3_files(path, self._gcd_rescue)
        assert len(i3_files) == len(gcd_files)

        filesets: List[I3FileSet] = []
        for i3_file, gcd_file in zip(i3_files, gcd_files):
            assert isinstance(i3_file, str)
            assert isinstance(gcd_file, str)
            filesets.append(I3FileSet(i3_file, gcd_file))

        self.validate_files(filesets)
        return filesets

    def _skip_frame(self, frame: "icetray.I3Frame") -> bool:  # type: ignore[name-defined]
        """Return True if frame should be skipped according to filters."""
        if self._i3filters is None:
            return False
        for f in self._i3filters:
            if not f(frame):
                return True
        return False
