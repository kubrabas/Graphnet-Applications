from typing import TYPE_CHECKING, Any, Dict, List, Optional
from graphnet.data.extractors.icecube.utilities.frames import (get_om_keys_and_pulseseries)
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray, dataclasses, dataio  # pyright: reportMissingImports=false

from graphnet.data.extractors.icecube import (
    I3FeatureExtractorIceCube86,
    I3FeatureExtractor,
    I3Extractor
)


class I3FeatureExtractorPONE(I3FeatureExtractorIceCube86):
    """Class for extracting reconstructed features for P-ONE events created with LeptonInjector."""
    
    def __init__(
        self,
        pulsemap: str,
        name: str = "feature", 
        exclude: list = [None],
    ):
       
        # Base class constructor
        super().__init__(pulsemap=pulsemap, exclude=exclude)
        self._extractor_name = name

        self._detector_status: Optional["icetray.I3Frame.DetectorStatus"] = None 
        
        
        
    def set_gcd(self, i3_file: str, gcd_file: Optional[str] = None) -> None:
        """Extract GFrame, CFrame and DFrame from i3/gcd-file pair.

           Information from these frames will be set as member variables of
           `I3Extractor.`

        Args:
            i3_file: Path to i3 file that is being converted.
            gcd_file: Path to GCD file. Defaults to None. If no GCD file is
                      given, the method will attempt to find C and G frames in
                      the i3 file instead. If either one of those are not
                      present, `RuntimeErrors` will be raised.
        """
        super().set_gcd(i3_file=i3_file, gcd_file=gcd_file)
        
        if gcd_file is None:
            # If no GCD file is provided, search the I3 file for frames
            # containing geometry (GFrame) and calibration (CFrame)
            gcd = dataio.I3File(i3_file)
        else:
            # Ideally ends here
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
        """Extract reconstructed features from `frame`.

        Args:
            frame: Physics (P) I3-frame from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """
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

        # Added these :
        is_bright_dom = -1
        is_saturated_dom = -1
        is_errata_dom = -1
        
        bad_doms = None
        
        if "BadDomsList" in self._detector_status:
            bad_doms = self._detector_status["BadDomsList"]


        event_time = frame["I3EventHeader"].start_time.mod_julian_day_double  ## what is this

        for om_key in om_keys:
            # Common values for each OM
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

            # DOM flags

            if bad_doms:
                is_bad_dom = 1 if om_key in bad_doms else 0
            else:
                is_bad_dom = int(padding_value)


            # Loop over pulses for each OM
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
                # ID's
                output["string"].append(string)
                output["pmt_number"].append(pmt_number)
                output["dom_number"].append(dom_number)
                output["dom_type"].append(dom_type)
                # DOM flags
                output["is_bad_dom"].append(is_bad_dom)
                output["event_time"].append(event_time)
                output["is_bright_dom"].append(is_bright_dom)
                output["is_errata_dom"].append(is_errata_dom)
                output["is_saturated_dom"].append(is_saturated_dom)

                # Pulse flags
                flags = getattr(pulse, "flags", padding_value)
                if flags == padding_value:
                    output["hlc"].append(padding_value)
                    output["awtd"].append(padding_value)
                else:
                    output["hlc"].append((pulse.flags >> 0) & 0x1)  # bit 0
                    output["awtd"].append(self._parse_awtd_flag(pulse))

        return output
    



 # icecube86'den editlemem gereken method var mi bak. buradaki yaptiklarim duzenli mi ona da bak. temizle bi her seyi. 
 # time seylerini anla
  