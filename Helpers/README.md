not up-to-date

### I3Compressor

Utility for compressing IceCube `.i3` files to `.i3.gz`.

```python
from Helpers.extensions import I3Compressor

# Keep original .i3 files
compressor = I3Compressor(delete_original=False)
compressor.compress("/project/def-nahee/kbas/POM_Response")

# Or: compress a single file and delete the original
compressor = I3Compressor(delete_original=True)
compressor.compress("/project/def-nahee/kbas/POM_Response/pom_response_batch_4911.i3")
```

### FrameKeyToTable

Utility for turning frame objects (e.g. `EventProperties`, `I3EventHeader`) from
Physics frames into a pandas `DataFrame`.

```python
from Helpers.tabulators import FrameKeyToTable

DATA_PATH = "/project/def-nahee/kbas/POM_Response/pom_response_batch_000.i3.gz"

# Example 1: EventProperties → DataFrame
ep_table = FrameKeyToTable(
    data_path=DATA_PATH,
    frame_key="EventProperties",
)
df_ep = ep_table.to_dataframe(max_events=300)
df_ep

# Example 2: I3EventHeader → DataFrame
i3eh_table = FrameKeyToTable(
    data_path=DATA_PATH,
    frame_key="I3EventHeader",
)
df_i3eh = i3eh_table.to_dataframe(max_events=50)
df_i3eh
```





