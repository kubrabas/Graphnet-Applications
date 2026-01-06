#!/usr/bin/env python3
"""
DynEdge energy reconstruction (GraphNeT) on P-ONE parquet data.

- Parquet layout expected:
  PARQUET_ROOT/
    truth/truth_<chunk_id>.parquet
    features/features_<chunk_id>.parquet

- Splitting is done by unique event_no across ALL truth parquet files.
- Energy target follows DynEdge paper:
    y = log10(E_true/GeV)
  (your energy is already in GeV, so y = log10(energy))

- Loss: LogCosh on residual in log10 space (paper).
- LR schedule: piecewise linear one-cycle style (paper).
"""

from __future__ import annotations

import os
import re
import json
import math
from glob import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import polars as pol
import torch
from torch import Tensor
from torch.utils.data import Subset
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# -----------------------------
# GraphNeT imports (your provided source matches these)
# -----------------------------
from graphnet.data.dataset import ParquetDataset
from graphnet.data.dataloader import DataLoader as GraphNeTDataLoader

from graphnet.models.data_representation import GraphDefinition
from graphnet.models.data_representation.graphs.edges.edges import KNNEdges
from graphnet.models.data_representation.graphs.nodes.nodes import NodesAsPulses

from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.task import IdentityTask

from graphnet.training.loss_functions import LogCoshLoss
from graphnet.training.labels import Label

from graphnet.training.callbacks import (
    ProgressBar,
    GraphnetEarlyStopping,
    PiecewiseLinearLR,
)

# =====================================================
# USER CONFIG
# =====================================================

PARQUET_ROOT: str = "/project/def-nahee/kbas/POM_Response_Parquet/merged"

PULSEMAP_TABLE: str = "features"
TRUTH_TABLE: str = "truth"

INDEX_COLUMN: str = "event_no"

FEATURES: List[str] = ["dom_x", "dom_y", "dom_z", "dom_time", "charge"]

# Truth columns we need in the graph object
TRUTH_COLUMNS: List[str] = ["energy"]

# Split
SEED: int = 42
TRAIN_FRACTION: float = 0.80
VAL_FRACTION: float = 0.10
TEST_FRACTION: float = 0.10  # should satisfy TRAIN+VAL+TEST=1

# Robust scaling (fit on TRAIN only)
ROBUST_SAMPLE_MAX_EVENTS: int = 50_000   # cap events used for scaler fit (printy + safe)
ROBUST_SAMPLE_MAX_PULSES_PER_EVENT: int = 512  # cap pulses per event used for scaler fit
ROBUST_EPS: float = 1e-12

# DataLoader
BATCH_SIZE: int = 1024
SHUFFLE_TRAIN: bool = True
SHUFFLE_VAL: bool = False
SHUFFLE_TEST: bool = False
NUM_WORKERS: int = 8
PERSISTENT_WORKERS: bool = True
PREFETCH_FACTOR: int = 2
PIN_MEMORY: bool = True
MULTIPROCESSING_CONTEXT: str = "spawn"

# Training (paper: 30 epochs, early stopping patience 5)
MAX_EPOCHS: int = 30
EARLY_STOPPING_PATIENCE: int = 5

# LR schedule (paper)
BASE_LR: float = 1e-3
MIN_LR: float = 1e-5  # start/end
WARMUP_FRACTION_OF_FIRST_EPOCH: float = 0.50  # first epoch’s first 50% iterations

# Output
OUTPUT_DIR: str = "./dynedge_energy_pone_run"
SCALER_PATH: str = os.path.join(OUTPUT_DIR, "robust_scaler.json")


# =====================================================
# HELPERS
# =====================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _discover_chunk_files(root: str, table: str) -> Dict[int, str]:
    """
    Expect files like: {root}/{table}/{table}_{chunk_id}.parquet
    Return dict: chunk_id -> path
    """
    pattern = os.path.join(root, table, f"{table}_*.parquet")
    files = glob(pattern)
    if len(files) == 0:
        raise FileNotFoundError(f"No parquet files found with pattern: {pattern}")

    rx = re.compile(rf"{re.escape(table)}_(\d+)\.parquet$")
    out: Dict[int, str] = {}
    for fp in files:
        m = rx.search(fp)
        if m is None:
            continue
        out[int(m.group(1))] = fp

    if len(out) == 0:
        raise FileNotFoundError(
            f"Found parquet files but none match naming {table}_<id>.parquet in: {os.path.join(root, table)}"
        )

    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _collect_event_ids_in_file(truth_fp: str, index_column: str) -> np.ndarray:
    df = pol.read_parquet(truth_fp).select([index_column])
    # ParquetDataset sorts by index_column; match that for sequential index mapping:
    df = df.sort(index_column)
    return df[index_column].to_numpy()


def _split_event_ids(
    all_event_ids: np.ndarray,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs((train_fraction + val_fraction + test_fraction) - 1.0) < 1e-9

    rng = np.random.default_rng(seed=seed)
    ids = all_event_ids.copy()
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(math.floor(train_fraction * n))
    n_val = int(math.floor(val_fraction * n))
    n_test = n - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    assert len(train_ids) + len(val_ids) + len(test_ids) == n
    assert len(test_ids) == n_test

    return train_ids, val_ids, test_ids


def _event_ids_to_sequential_indices(
    truth_files: Dict[int, str],
    index_column: str,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Build sequential indices used by ParquetDataset.__getitem__:
      sequential_index = offset(chunk_id) + row_in_sorted_truth_file
    """
    train_set = set(map(int, train_ids.tolist()))
    val_set = set(map(int, val_ids.tolist()))
    test_set = set(map(int, test_ids.tolist()))

    train_seq: List[int] = []
    val_seq: List[int] = []
    test_seq: List[int] = []

    offset = 0
    for chunk_id, truth_fp in truth_files.items():
        event_ids = _collect_event_ids_in_file(truth_fp, index_column=index_column)
        # event_ids are sorted by event_no to match ParquetDataset internal sort
        for local_row, ev in enumerate(event_ids.tolist()):
            ev_int = int(ev)
            seq = offset + local_row
            if ev_int in train_set:
                train_seq.append(seq)
            elif ev_int in val_set:
                val_seq.append(seq)
            elif ev_int in test_set:
                test_seq.append(seq)
        offset += len(event_ids)

    print(f"[split->sequential] train_seq={len(train_seq)} val_seq={len(val_seq)} test_seq={len(test_seq)} total={offset}")

    return train_seq, val_seq, test_seq


# =====================================================
# LABEL: log10(energy/GeV)
# =====================================================

class Log10Energy(Label):
    def __init__(
        self,
        key: str = "energy_log10",
        energy_key: str = "energy",
    ):
        super().__init__(key=key)
        self._energy_key = energy_key

    def __call__(self, graph) -> torch.tensor:
        e = graph[self._energy_key]
        # handle shapes [1] or [1,1]
        e = e.reshape(-1).to(torch.float32)
        # Energy already in GeV → log10(E/GeV) == log10(E)
        return torch.log10(e + 1e-24).reshape(1)


# =====================================================
# ROBUST SCALER + DETECTOR
# =====================================================

@dataclass
class RobustScalerParams:
    median: List[float]
    iqr: List[float]
    feature_names: List[str]


def _fit_robust_scaler_from_dataset(
    dataset: ParquetDataset,
    train_seq_indices: List[int],
    feature_names: List[str],
    max_events: int,
    max_pulses_per_event: int,
    eps: float,
) -> RobustScalerParams:
    print("[robust-scaler] Fitting robust scaler on TRAIN only...")
    print(f"[robust-scaler] max_events={max_events} max_pulses_per_event={max_pulses_per_event}")

    rng = np.random.default_rng(seed=SEED)
    chosen = train_seq_indices.copy()
    rng.shuffle(chosen)
    chosen = chosen[: min(len(chosen), max_events)]

    samples: List[np.ndarray] = []

    for i, seq in enumerate(chosen):
        if (i + 1) % 1000 == 0:
            print(f"[robust-scaler] processed_events={i+1}/{len(chosen)}")

        g = dataset[seq]
        x = g.x.detach().cpu().numpy()  # shape [n_pulses, n_features]
        if x.shape[0] > max_pulses_per_event:
            idx = rng.choice(x.shape[0], size=max_pulses_per_event, replace=False)
            x = x[idx]
        samples.append(x)

    X = np.concatenate(samples, axis=0)
    print(f"[robust-scaler] total_pulses_used={X.shape[0]}  features={X.shape[1]}")

    med = np.median(X, axis=0)
    q25 = np.quantile(X, 0.25, axis=0)
    q75 = np.quantile(X, 0.75, axis=0)
    iqr = np.maximum(q75 - q25, eps)

    return RobustScalerParams(
        median=med.tolist(),
        iqr=iqr.tolist(),
        feature_names=list(feature_names),
    )


def _save_scaler(params: RobustScalerParams, path: str) -> None:
    payload = {
        "feature_names": params.feature_names,
        "median": params.median,
        "iqr": params.iqr,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[robust-scaler] saved -> {path}")


def _load_scaler(path: str) -> RobustScalerParams:
    with open(path, "r") as f:
        payload = json.load(f)
    return RobustScalerParams(
        feature_names=payload["feature_names"],
        median=payload["median"],
        iqr=payload["iqr"],
    )


# IMPORTANT:
# GraphNeT’s Detector base class differs across versions.
# You asked “Detector’u sen yaz” — so here is a minimal, explicit one.
#
# If your local GraphNeT already has an abstract Detector interface,
# update the inheritance/import accordingly (but keep the methods).



class PONE:
    def __init__(
        self,
        *,
        feature_names: List[str],
        scaler: Optional[RobustScalerParams] = None,
        scaler_path: Optional[str] = None,
        # --- geriye uyum: senin main() böyle çağırıyor ---
        scaler_params: Optional[RobustScalerParams] = None,
        string_index_name: str = "__unused__",
        # ------------------------------------------------
        eps: float = 1e-12,
        dtype: torch.dtype = torch.float32,
        **kwargs,  # başka sürüm/çağrı farkları için sessiz yut
    ) -> None:
        self.feature_names: List[str] = list(feature_names)
        self.eps: float = float(eps)
        self.dtype: torch.dtype = dtype

        # Dataset compatibility
        self.string_index_name: str = string_index_name

        # scaler_params alias
        if scaler is None and scaler_params is not None:
            scaler = scaler_params

        if scaler is None and scaler_path is not None:
            scaler = _load_scaler(scaler_path)

        ...



# =====================================================
# LIGHTNING MODULE (minimal, explicit)
# =====================================================

class DynEdgeEnergyLightningModule(pl.LightningModule):
    def __init__(
        self,
        *,
        backbone: DynEdge,
        task: IdentityTask,
        steps_per_epoch: int,
        max_epochs: int,
        base_lr: float,
        min_lr: float,
        warmup_fraction_first_epoch: float,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "task"])

        self.backbone = backbone
        self.task = task

        self._steps_per_epoch = int(steps_per_epoch)
        self._max_epochs = int(max_epochs)

        self._base_lr = float(base_lr)
        self._min_lr = float(min_lr)
        self._warmup_fraction_first_epoch = float(warmup_fraction_first_epoch)

        self._adam_beta1 = float(adam_beta1)
        self._adam_beta2 = float(adam_beta2)
        self._adam_eps = float(adam_eps)
        self._weight_decay = float(weight_decay)

    def forward(self, batch):
        z = self.backbone(batch)  # [batch_size, hidden]
        pred = self.task(z)       # [batch_size, 1]
        return pred

    def _shared_step(self, batch, stage: str) -> Tensor:
        pred = self.forward(batch)
        loss = self.task.compute_loss(pred, batch)

        # log10 target
        y = batch["energy_log10"].reshape(-1, 1).to(pred.dtype)
        mae_log10 = torch.mean(torch.abs(pred - y))

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=BATCH_SIZE)
        self.log(f"{stage}_mae_log10", mae_log10, prog_bar=True, on_step=False, on_epoch=True, batch_size=BATCH_SIZE)

        return loss

    def training_step(self, batch, batch_idx: int) -> Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx: int) -> None:
        _ = self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx: int) -> None:
        _ = self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            params=self.parameters(),
            lr=self._base_lr,
            betas=(self._adam_beta1, self._adam_beta2),
            eps=self._adam_eps,
            weight_decay=self._weight_decay,
            amsgrad=False,
        )

        # Paper schedule:
        # - Warmup in first epoch: lr 1e-5 -> 1e-3 over first 50% iterations of epoch 1
        # - Then linearly down to 1e-5 over remaining iterations in the 30-epoch budget
        warmup_steps = int(self._steps_per_epoch * self._warmup_fraction_first_epoch)
        total_steps = int(self._steps_per_epoch * self._max_epochs)

        # PiecewiseLinearLR multiplies base_lr by factors.
        # We want:
        #   base_lr * f0 = min_lr  => f0 = min_lr/base_lr
        #   base_lr * f1 = base_lr => f1 = 1
        #   base_lr * f2 = min_lr  => f2 = min_lr/base_lr
        f0 = self._min_lr / self._base_lr
        f1 = 1.0
        f2 = self._min_lr / self._base_lr

        sched = PiecewiseLinearLR(
            optimizer=opt,
            milestones=[0, warmup_steps, total_steps],
            factors=[f0, f1, f2],
            last_epoch=-1,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }


# =====================================================
# BUILD DATASET + GRAPHDEF
# =====================================================

def _build_graph_definition(detector: PONE) -> GraphDefinition:
    # Node definition: pulses are nodes, just use provided FEATURES
    node_def = NodesAsPulses(
        input_feature_names=FEATURES,
        # write explicit defaults (these may vary by GraphNeT version)
        position_feature_names=None,
        time_feature_names=None,
        charge_feature_names=None,
    )

    # Edge definition: kNN in xyz columns [0,1,2] (paper uses 8-NN)
    edge_def = KNNEdges(
        nb_nearest_neighbours=8,
        columns=[0, 1, 2],
    )

    graph_def = GraphDefinition(
        detector=detector,
        node_definition=node_def,
        edge_definition=edge_def,
        # explicit defaults (if your GraphNeT signature differs, keep these explicit)
        input_feature_names=FEATURES,
        dtype=torch.float32,
    )

    return graph_def


def _build_base_dataset(detector: PONE) -> ParquetDataset:
    graph_def = _build_graph_definition(detector=detector)

    ds = ParquetDataset(
        path=PARQUET_ROOT,
        pulsemaps=[PULSEMAP_TABLE],
        features=FEATURES,
        truth=TRUTH_COLUMNS,
        data_representation=graph_def,    # preferred
        graph_definition=None,            # explicit (deprecated path)
        node_truth=None,
        index_column=INDEX_COLUMN,
        truth_table=TRUTH_TABLE,
        node_truth_table=None,
        string_selection=None,
        selection=None,                  # all chunks
        dtype=torch.float32,
        loss_weight_table=None,
        loss_weight_column=None,
        loss_weight_default_value=None,
        seed=None,
        cache_size=1,
        labels=None,
    )

    # Add runtime label used for training target (log10 energy)
    ds.add_label(
        key="energy_log10",
        fn=Log10Energy(
            key="energy_log10",
            energy_key="energy",
        ),
    )

    return ds


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    _ensure_dir(OUTPUT_DIR)

    print("============================================================")
    print("[setup] Discovering parquet chunks...")
    truth_files = _discover_chunk_files(PARQUET_ROOT, TRUTH_TABLE)
    feat_files = _discover_chunk_files(PARQUET_ROOT, PULSEMAP_TABLE)
    print(f"[setup] truth_chunks={len(truth_files)} features_chunks={len(feat_files)}")

    missing = set(truth_files.keys()) ^ set(feat_files.keys())
    if len(missing) > 0:
        print(f"[warning] truth/features chunk_id mismatch. symmetric_diff={sorted(list(missing))}")

    print("============================================================")
    print("[setup] Collecting unique event_no across ALL truth files...")
    all_ids_list: List[np.ndarray] = []
    for chunk_id, fp in truth_files.items():
        ids = _collect_event_ids_in_file(fp, index_column=INDEX_COLUMN)
        all_ids_list.append(ids)
        print(f"[events] chunk_id={chunk_id} events_in_file={len(ids)}")

    all_event_ids = np.unique(np.concatenate(all_ids_list, axis=0))
    print(f"[events] total_unique_events={len(all_event_ids)}")

    print("============================================================")
    print("[split] Splitting by event_no (train/val/test)...")
    train_ids, val_ids, test_ids = _split_event_ids(
        all_event_ids=all_event_ids,
        seed=SEED,
        train_fraction=TRAIN_FRACTION,
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
    )
    print(f"[split] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    print("============================================================")
    print("[split] Mapping event_no -> sequential indices used by ParquetDataset...")
    train_seq, val_seq, test_seq = _event_ids_to_sequential_indices(
        truth_files=truth_files,
        index_column=INDEX_COLUMN,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )

    print("============================================================")
    print("[dataset] Building dataset (identity scaler first, to fit robust scaler on TRAIN)...")
    detector = PONE(
        feature_names=FEATURES,
        scaler_params=None,
        string_index_name="string",
    )
    dataset = _build_base_dataset(detector=detector)

    # Fit robust scaler on TRAIN only (unless already exists)
    if os.path.exists(SCALER_PATH):
        print(f"[robust-scaler] Found existing scaler at {SCALER_PATH} -> loading")
        params = _load_scaler(SCALER_PATH)
        detector.set_scaler(params)
    else:
        params = _fit_robust_scaler_from_dataset(
            dataset=dataset,
            train_seq_indices=train_seq,
            feature_names=FEATURES,
            max_events=ROBUST_SAMPLE_MAX_EVENTS,
            max_pulses_per_event=ROBUST_SAMPLE_MAX_PULSES_PER_EVENT,
            eps=ROBUST_EPS,
        )
        _save_scaler(params, SCALER_PATH)
        detector.set_scaler(params)

    print(f"[robust-scaler] detector.has_scaler={detector.has_scaler()}")

    print("============================================================")
    print("[dataset] Re-building dataset WITH robust scaler active...")
    dataset = _build_base_dataset(detector=detector)

    train_ds = Subset(dataset, train_seq)
    val_ds = Subset(dataset, val_seq)
    test_ds = Subset(dataset, test_seq)

    print(f"[dataset] train_len={len(train_ds)} val_len={len(val_ds)} test_len={len(test_ds)}")

    print("============================================================")
    print("[dataloader] Creating dataloaders...")
    train_loader = GraphNeTDataLoader(
        dataset=train_ds,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAIN,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        collate_fn=None,  # explicit: GraphNeTDataLoader has default collate_fn
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=PIN_MEMORY,
        multiprocessing_context=MULTIPROCESSING_CONTEXT,
        drop_last=False,
        timeout=0,
    )

    val_loader = GraphNeTDataLoader(
        dataset=val_ds,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_VAL,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        collate_fn=None,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=PIN_MEMORY,
        multiprocessing_context=MULTIPROCESSING_CONTEXT,
        drop_last=False,
        timeout=0,
    )

    test_loader = GraphNeTDataLoader(
        dataset=test_ds,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TEST,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        collate_fn=None,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=PIN_MEMORY,
        multiprocessing_context=MULTIPROCESSING_CONTEXT,
        drop_last=False,
        timeout=0,
    )

    steps_per_epoch = len(train_loader)
    print(f"[dataloader] steps_per_epoch={steps_per_epoch}")

    print("============================================================")
    print("[model] Building DynEdge backbone (paper-like defaults, explicit)...")
    backbone = DynEdge(
        nb_inputs=len(FEATURES),
        nb_neighbours=8,
        features_subset=slice(0, 3),
        dynedge_layer_sizes=[(128, 256), (336, 256), (336, 256), (336, 256)],
        post_processing_layer_sizes=[336, 256],
        readout_layer_sizes=[128],
        global_pooling_schemes=["min", "max", "mean", "sum"],
        add_global_variables_after_pooling=False,
        activation_layer="relu",
        add_norm_layer=False,
        skip_readout=False,
    )

    print("============================================================")
    print("[task] Building energy task in log10 space + LogCoshLoss (paper)...")
    loss_fn = LogCoshLoss(
        # explicit defaults: LossFunction(Model) takes **kwargs, keep empty
    )

    task = IdentityTask(
        nb_outputs=1,
        target_labels="energy_log10",
        hidden_size=backbone.nb_outputs,
        loss_function=loss_fn,
        prediction_labels=["energy_log10_pred"],
        transform_prediction_and_target=None,
        transform_target=None,
        transform_inference=None,
        transform_support=None,
        loss_weight=None,
    )

    lit_model = DynEdgeEnergyLightningModule(
        backbone=backbone,
        task=task,
        steps_per_epoch=steps_per_epoch,
        max_epochs=MAX_EPOCHS,
        base_lr=BASE_LR,
        min_lr=MIN_LR,
        warmup_fraction_first_epoch=WARMUP_FRACTION_OF_FIRST_EPOCH,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,
    )

    print("============================================================")
    print("[trainer] Configuring callbacks (paper-like early stopping)...")
    callbacks = [
        ProgressBar(refresh_rate=1),
        GraphnetEarlyStopping(
            save_dir=OUTPUT_DIR,
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            mode="min",
            verbose=True,
            check_on_train_epoch_end=False,
            strict=True,
            min_delta=0.0,
        ),
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    trainer = Trainer(
        default_root_dir=OUTPUT_DIR,
        accelerator=accelerator,
        devices=devices,
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        log_every_n_steps=50,
        enable_checkpointing=False,  # GraphnetEarlyStopping saves best_model.pth
        enable_model_summary=True,
        deterministic=True,
        benchmark=False,
    )

    print("============================================================")
    print("[train] Starting training...")
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    print("============================================================")
    print("[validate] Running validation (best model already restored by GraphnetEarlyStopping.on_fit_end)...")
    val_metrics = trainer.validate(
        model=lit_model,
        dataloaders=val_loader,
        verbose=True,
    )
    print(f"[validate] metrics={val_metrics}")

    print("============================================================")
    print("[test] Running test...")
    test_metrics = trainer.test(
        model=lit_model,
        dataloaders=test_loader,
        verbose=True,
    )
    print(f"[test] metrics={test_metrics}")

    print("============================================================")
    print("[done] Outputs in:", OUTPUT_DIR)
    print(" - robust scaler:", SCALER_PATH)
    print(" - best model state_dict:", os.path.join(OUTPUT_DIR, "best_model.pth"))
    print("============================================================")


if __name__ == "__main__":
    main()
