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
