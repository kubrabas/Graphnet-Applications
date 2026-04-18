# IceTray Environment (Fir Cluster)

1. Allocate a GPU node:
   ```bash
   salloc --time=0:30:00 --account=def-nahee --mem=48G --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
   ```
2. Load modules:
   ```bash
   module load StdEnv/2020 gcc/11.3.0 apptainer scipy-stack/2023b
   ```
3. Enter the IceTray container:
   ```bash
   apptainer shell --nv /cvmfs/software.pacific-neutrino.org/containers/itray_v1.17.1
   ```
4. Set up the IceTray environment inside the container:
   ```bash
   source /usr/local/icetray/build/env-shell.sh
   ```

**Note:** Haven't tested this with pone-offline. Also haven't checked the contents of `env-shell.sh`.


---
