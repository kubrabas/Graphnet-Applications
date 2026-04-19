# IceTray Environment (Fir Cluster)

1. Allocate a GPU node:
   ```bash
   salloc --time=1:30:00 --account=def-nahee --mem=24G --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
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

# VS Code - Interactive Jupyter Notebook (IceTray kernel)

The scripts in `~/slurm_scripts/` automate steps 1–4 above and start a Jupyter server on the compute node.

**Terminal 1** - start Jupyter:
```bash
bash ~/slurm_scripts/start_icetray_jupyter.sh
```
Wait until it prints the node name, port, and a URL with a token.

**Terminal 2** - open SSH tunnel (use the node/port printed by Terminal 1):
```bash
bash ~/slurm_scripts/tunnel_to_node.sh <node> <port>
# e.g. bash ~/slurm_scripts/tunnel_to_node.sh fc11020 8888
```

**VS Code** - connect to the Jupyter server:
1. Open a `.ipynb` file
2. Click the kernel selector (top right) → **Select Kernel** → **Existing Jupyter Server**
3. Paste the URL printed by Terminal 1 (includes the token)
4. Choose the **Python 3 (ipykernel)** kernel — this is the IceTray environment

Press Ctrl+C in Terminal 1 to shut everything down.
