# IceTray Environment (Fir Cluster)

1. Allocate a GPU node:
   ```bash
   salloc --time=2:30:00 --account=def-nahee --mem=24G --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
   ```
2. Load modules:
   ```bash
   module load StdEnv/2020 gcc/11.3.0 apptainer scipy-stack/2023b
   ```
3. Enter the IceTray container:
   ```bash
   apptainer shell --nv -B /cvmfs/software.pacific-neutrino.org/ /cvmfs/software.pacific-neutrino.org/containers/itray_v1.17.1
   ```
4. Set up the IceTray environment inside the container:
   ```bash
   source /usr/local/icetray/build/env-shell.sh
   ```

   for LeptonWeighter:
   ```bash
   export PYTHONPATH=/usr/local/LeptonWeighter/lib:$PYTHONPATH
   export LD_LIBRARY_PATH=/usr/local/LeptonWeighter/lib:/usr/local/lib:$LD_LIBRARY_PATH
   ```

   for pone_offline (e.g. PMT response notebooks):
   ```bash
   export PYTHONPATH=/cvmfs/software.pacific-neutrino.org/pone_offline/v2.0:$PYTHONPATH
   export PONESRCDIR=/project/6008051/pone_simulation/pone_offline
   ```

**Note:** Haven't checked the contents of `env-shell.sh`.


---

# VS Code — Interactive Jupyter Notebook (IceTray Kernel)

To use Jupyter notebook in VS Code with the IceTray environment set up above, follow these steps after completing the four setup steps given above.

1. Start a Jupyter server on the compute node:
   ```bash
   jupyter notebook --no-browser --ip=127.0.0.1 --port=8888
   ```

2. Copy the token URL printed in the output, e.g.:
   ```
   http://127.0.0.1:8888/?token=2c893adc2ffaa631d643200174bcd35072402dff09a25fc0
   ```

3. In a **separate terminal**, open an SSH tunnel to the compute node (replace `fc11020` with your actual node name):
   ```bash
   ssh -N -L 8888:localhost:8888 fc11020
   ```
   > Nothing will appear in this terminal. that is expected.

4. In VS Code, open a Jupyter notebook, click **Select Kernel** → **Existing Jupyter Server**, paste the URL from step 2, and give the server a name. Then select **Python 3 (ipykernel)**.

---

# Monitoring Slurm Jobs

To check the status of your submitted jobs on the cluster:

```bash
squeue -u kbas --format="%.18i %.12P %.50j %.8u %.8T %.10M %.10l %.6D %R"
```

This lists all jobs for user `kbas` with the following columns:

| Column | Description |
|--------|-------------|
| `JOBID` | Unique job ID assigned by Slurm |
| `PARTITION` | The partition (queue) the job was submitted to |
| `NAME` | Job name (up to 50 characters) |
| `USER` | Username who submitted the job |
| `STATE` | Current job state (e.g. `PENDING`, `RUNNING`, `COMPLETED`) |
| `TIME` | How long the job has been running |
| `TIME_LIMIT` | Maximum allowed runtime |
| `NODES` | Number of nodes allocated |
| `NODELIST(REASON)` | Node(s) the job runs on, or the reason it is still pending |
