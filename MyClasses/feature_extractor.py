from typing import TYPE_CHECKING, Any, Dict, List, Optional
from graphnet.data.extractors.icecube.utilities.frames import (get_om_keys_and_pulseseries)
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray, dataclasses, dataio  # pyright: reportMissingImports=false

from graphnet.data.extractors.icecube import (
    I3FeatureExtractorIceCube86
)
import numpy as np



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
        self._PMT_ANGLES = np.array( [[57.5, 270.],  # PMT 1
                        [57.5, 0.],    # PMT 2
                        [57.5, 90.],   # PMT 3
                        [57.5, 180.],  # PMT 4
                        [25., 225.],   # PMT 5
                        [25., 315.],   # PMT 6
                        [25., 45.],    # PMT 7
                        [25., 135.],   # PMT 8
                        [-57.5, 270.], # PMT 9
                        [-57.5, 180.], # PMT 10
                        [-57.5, 90.],  # PMT 11
                        [-57.5, 0.],   # PMT 12
                        [-25., 315.],  # PMT 13
                        [-25., 225.],  # PMT 14
                        [-25., 135.],  # PMT 15
                        [-25., 45.]    # PMT 16
                        ])
        self.MODULE_RADIUS_M = 0.2159
        
        self._pmt_x_coordinates_wrt_om_rotated = np.multiply(np.sin(np.deg2rad(90. - self._PMT_ANGLES[:, 0])), np.cos(np.deg2rad(self._PMT_ANGLES[:, 1])))
        self._pmt_y_coordinates_wrt_om_rotated = np.multiply(np.sin(np.deg2rad(90. - self._PMT_ANGLES[:, 0])), np.sin(np.deg2rad(self._PMT_ANGLES[:, 1])))
        self._pmt_z_coordinates_wrt_om_rotated = np.cos(np.deg2rad(90. - self._PMT_ANGLES[:, 0]))
        
        self._PMT_MATRIX_rotated = np.array([self._pmt_x_coordinates_wrt_om_rotated,  self._pmt_y_coordinates_wrt_om_rotated, self._pmt_z_coordinates_wrt_om_rotated]).T
  
        self._PMT_COORDINATES_rotated = self._PMT_MATRIX_rotated * self.MODULE_RADIUS_M


        self._minus_90_degree_rotation_around_x_axis =  np.array([
            [1.000000e+00, 0.000000e+00, 0.000000e+00],
            [0.000000e+00, 0.000000e+00, 1.000000e+00],
            [0.000000e+00, -1.000000e+00, 0.000000e+00]], dtype=float)
    
    
        self._PMT_COORDINATES_ORIGINAL = self._PMT_COORDINATES_rotated @ self._minus_90_degree_rotation_around_x_axis.T
        
        ### kontrol et kankiiiii dogru mu burdaki islemler. görsel ile tekrar incele. 
        ### he bi de su silindiri rotate etme mevzusunu da bi incele
        ### pmt numaralari da karisiyor muuu ona da bak kankitom
        ### bi de init et ve kontrol et pmt konumlari dogru update edilmis mi OM'den
        

        
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
            "pmt_x": [],
            "pmt_y": [],
            "pmt_z": []
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
            
            pmt_x = pmt_y = pmt_z = padding_value
            
            if pmt_number is not None:
                idx = int(pmt_number) - 1
                if 0 <= idx < len(self._PMT_COORDINATES_ORIGINAL):
                    rel = self._PMT_COORDINATES_ORIGINAL[idx]  # rel = [dx, dy, dz]
                    pmt_x = x + float(rel[0])
                    pmt_y = y + float(rel[1])
                    pmt_z = z + float(rel[2])


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
                output["pmt_x"].append(pmt_x)
                output["pmt_y"].append(pmt_y)
                output["pmt_z"].append(pmt_z)
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
  