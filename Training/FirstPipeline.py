#### change this. dont use StandardModel


## 0) Imports and Little Helper Function:

import os
from datetime import datetime
from typing import Dict, Callable

import torch
from graphnet.models.detector.detector import Detector

from graphnet.models.data_representation import KNNGraph
from graphnet.models.data_representation.graphs import ClusterSummaryFeatures

from graphnet.models.gnn.dynedge import DynEdge
from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset
from graphnet.data.dataloader import DataLoader
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.models import StandardModel


def section_banner(name: str) -> None:
    line = "=" * 90
    print("\n" + line, flush=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ==> {name}", flush=True)
    print(line + "\n", flush=True)


section_banner("0) Imports and Helper")


## 0) Global Settings:

section_banner("0) Global Settings")

PARQUET_ROOT = "/project/def-nahee/kbas/POM_Response_Parquet/merged"
PULSEMAPS = "features"   # pulse/hit parquet folder
TRUTH_TABLE = "truth"    # truth parquet folder

TRUTH_LABELS = ["energy"]


## 0) Environment Information:

section_banner("0) Environment Information")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.set_float32_matmul_precision("medium")
    print("Set torch matmul precision to: medium")
else:
    print("GPU: (not available)")

print("os.cpu_count:", os.cpu_count())

if hasattr(os, "sched_getaffinity"):
    affinity = sorted(os.sched_getaffinity(0))
    print("affinity cores:", len(affinity))
    print("affinity set (first 20):", affinity[:20])

print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK"))
print("SLURM_JOB_CPUS_PER_NODE:", os.environ.get("SLURM_JOB_CPUS_PER_NODE"))

## 0) My Classes: PONE (Detector)

section_banner("0) My Classes: PONE (Detector)")


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
        return x  # x / 500.0

    def _time(self, x: torch.Tensor) -> torch.Tensor:
        return x  # (x - 1.0e4) / 3.0e4

    def _charge(self, x: torch.Tensor) -> torch.Tensor:
        return x  # torch.log10(x)


## 1) Data Representation (Graph Definition)

section_banner("1) Data Representation (Graph Definition)")

FEATURES = ["dom_x", "dom_y", "dom_z", "dom_time", "charge"]
detector = PONE(replace_with_identity=FEATURES)

K = 8
print(f"KNN configuration: K={K} (node = OM via ClusterSummaryFeatures)")

node_definition = ClusterSummaryFeatures(
    cluster_on=["dom_x", "dom_y", "dom_z"],
    input_feature_names=FEATURES,
    charge_label="charge",
    time_label="dom_time",
    add_counts=True,
)


data_representation = KNNGraph(
    detector=detector,
    node_definition=node_definition,
    nb_nearest_neighbours=K,
    columns=[0, 1, 2],
)

## 2) Dataset (Parquet -> PyG Data)

section_banner("2) Dataset (Parquet -> PyG Data)")

dataset_kwargs = dict(
    path=str(PARQUET_ROOT),
    pulsemaps=PULSEMAPS,
    truth_table=TRUTH_TABLE,
    features=FEATURES,
    truth=TRUTH_LABELS,
)

print("Dataset config:")
print("  path:", dataset_kwargs["path"])
print("  pulsemaps:", dataset_kwargs["pulsemaps"])
print("  truth_table:", dataset_kwargs["truth_table"])
print("  features:", dataset_kwargs["features"])
print("  truth:", dataset_kwargs["truth"])

dataset = ParquetDataset(**dataset_kwargs, data_representation=data_representation)
print("Dataset created.")

## 3) DataLoader

section_banner("3) DataLoader")

BATCH_SIZE = 4
NUM_WORKERS = 0 #2 #4

print(f"DataLoader config: batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}, shuffle=True")

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
)


## 4) Backbone

section_banner("4) Backbone")

print("Backbone: DynEdge")
backbone = DynEdge(
    nb_inputs=data_representation.nb_outputs,
    global_pooling_schemes=["min", "max", "mean"],
)


## 5) Task: Energy Reconstruction (event-level)

section_banner("5) Task: Energy Reconstruction")

ENERGY_LABEL = "energy"
print(f"Task: EnergyReconstruction (target_labels={[ENERGY_LABEL]})")

task = EnergyReconstruction(
    hidden_size=backbone.nb_outputs,
    target_labels=[ENERGY_LABEL],
    loss_function=LogCoshLoss(),
)

## 6) StandardModel

section_banner("6) StandardModel")

print("Building StandardModel ...")
model = StandardModel(
    tasks=[task],
    data_representation=data_representation,
    backbone=backbone,
)
print("Model created.")


## 7) Train

section_banner("7) Train")

MAX_EPOCHS = 5
GPUS = 1 if torch.cuda.is_available() else None
print(f"Starting training: max_epochs={MAX_EPOCHS}, gpus={GPUS}")

model.fit(
    train_dataloader=train_loader,
    val_dataloader=None,
    max_epochs=MAX_EPOCHS,
    gpus=GPUS,
    distribution_strategy="auto",
)
