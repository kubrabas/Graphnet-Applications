from typing import Dict, Callable
import torch
from graphnet.models.detector.detector import Detector

class PONE(Detector):
    """Detector class for P-ONE."""


    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        return {
            "dom_x": self._xyz,
            "dom_y": self._xyz,
            "dom_z": self._xyz,
            "dom_time": self._time,
            "charge": self._charge,
        }

    def _xyz(self, x: torch.Tensor) -> torch.Tensor:
        return x     # x / 500.0   

    def _time(self, x: torch.Tensor) -> torch.Tensor:
        return x     # (x - 1.0e4) / 3.0e4  # şimdilik; sonra ayarlarız

    def _charge(self, x: torch.Tensor) -> torch.Tensor:
        return x     # torch.log10(x)


