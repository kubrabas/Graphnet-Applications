from icecube import dataio, icetray , dataclasses 
import pandas as pd
pd.set_option("display.max_columns", None)   
from collections.abc import Iterable

GCD_PATH = "/scratch/kbas/102_string/GCD_102strings.i3.gz"
data_file = dataio.I3File(GCD_PATH)
geometry = data_file.pop_frame()

print(type(geometry))
print("Stop:", geometry.Stop)   
print("Key's:", list(geometry.keys()))


om_map = geometry["I3OMGeoMap"]
print(type(om_map))
print("Number of OMs:", len(om_map))

# 102 * 20 * 16 = 32640


# print(om_map)


# unique string IDs (sorted)
unique_strings = sorted({omkey.string for omkey in om_map.keys()})

print("N_unique_strings:", len(unique_strings))
print("unique_strings:", unique_strings)

# comma-separated (tek satır)
print("csv:", ",".join(map(str, unique_strings)))