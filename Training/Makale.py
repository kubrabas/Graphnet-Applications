#!/usr/bin/env python3
"""
DynEdge energy reconstruction (GraphNeT) on P-ONE parquet data.

Target:
  y = log10(E_true/GeV)  (energy already in GeV -> y = log10(energy))

Loss:
  LogCosh in log10 space (via GraphNeT IdentityTask + LogCoshLoss)

LR schedule:
  PiecewiseLinearLR warmup then decay (paper-like)

Notes (fixes your hang):
- Default NUM_WORKERS=0 (most common cause of “Sanity Checking” hang on clusters is dataloader workers).
- Disable Lightning sanity check by default (num_sanity_val_steps=0).
- Guard warmup_steps so milestones are strictly increasing (steps_per_epoch can be 1 in your setup).
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
# GraphNeT imports
# -----------------------------
from graphnet.data.dataset import ParquetDataset
from graphnet.models.data_representation import GraphDefinition
from graphnet.models.data_representation.graphs.edges.edges import KNNEdges
from graphnet.models.data_representation.graphs.nodes.nodes import NodesAsPulses
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.task import IdentityTask
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.training.labels import Label
from graphnet.training.callbacks import ProgressBar, GraphnetEarlyStopping, PiecewiseLinearLR

# IMPORTANT: PyG DataLoader collates torch_geometric.data.Data correctly
from torch_geometric.loader import DataLoader as PyGDataLoader

# =====================================================
# USER CONFIG
# =====================================================

PARQUET_ROOT: str = "/project/def-nahee/kbas/POM_Response_Parquet/merged"

PULSEMAP_TABLE: str = "features"
TRUTH_TABLE: str = "truth"
INDEX_COLUMN: str = "event_no"

FEATURES: List[str] = ["dom_x", "dom_y", "dom_z", "dom_time", "charge"]
TRUTH_COLUMNS: List[str] = ["energy"]

SEED: int = 42
TRAIN_FRACTION: float = 0.80
VAL_FRACTION: float = 0.10
TEST_FRACTION: float = 0.10

ROBUST_SAMPLE_MAX_EVENTS: int = 50_000
ROBUST_SAMPLE_MAX_PULSES_PER_EVENT: int = 512
ROBUST_EPS: float = 1e-12

BATCH_SIZE: int = 4
SHUFFLE_TRAIN: bool = True
SHUFFLE_VAL: bool = False
SHUFFLE_TEST: bool = False

# ---- Critical stability defaults ----
# If you want workers later: export NUM_WORKERS=2 (or 4) and re-run.
NUM_WORKERS: int = int(os.environ.get("NUM_WORKERS", "0"))
PERSISTENT_WORKERS: bool = bool(int(os.environ.get("PERSISTENT_WORKERS", "0")))  # default False
PREFETCH_FACTOR: int = int(os.environ.get("PREFETCH_FACTOR", "2"))
PIN_MEMORY: bool = bool(int(os.environ.get("PIN_MEMORY", "0")))  # default False
# -------------------------------------

MAX_EPOCHS: int = 30
EARLY_STOPPING_PATIENCE: int = 5

BASE_LR: float = 1e-3
MIN_LR: float = 1e-5
WARMUP_FRACTION_OF_FIRST_EPOCH: float = 0.50

OUTPUT_DIR: str = "./dynedge_energy_pone_run"
SCALER_PATH: str = os.path.join(OUTPUT_DIR, "robust_scaler.json")

# Lightning sanity check can hang if val loader hangs; default OFF.
NUM_SANITY_VAL_STEPS: int = int(os.environ.get("NUM_SANITY_VAL_STEPS", "0"))

# Optional: quick debug to force-fetch one batch before Trainer starts.
DEBUG_FETCH_ONE_BATCH: bool = bool(int(os.environ.get("DEBUG_FETCH_ONE_BATCH", "1")))


# =====================================================
# HELPERS
# =====================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _discover_chunk_files(root: str, table: str) -> Dict[int, str]:
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
            f"Found parquet files but none match {table}_<id>.parquet in: {os.path.join(root, table)}"
        )

    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _collect_event_ids_in_file(truth_fp: str, index_column: str) -> np.ndarray:
    df = pol.read_parquet(truth_fp).select([index_column]).sort(index_column)
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

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


def _event_ids_to_sequential_indices(
    truth_files: Dict[int, str],
    index_column: str,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
) -> Tuple[List[int], List[int], List[int]]:
    train_set = set(map(int, train_ids.tolist()))
    val_set = set(map(int, val_ids.tolist()))
    test_set = set(map(int, test_ids.tolist()))

    train_seq: List[int] = []
    val_seq: List[int] = []
    test_seq: List[int] = []

    offset = 0
    for _, truth_fp in truth_files.items():
        event_ids = _collect_event_ids_in_file(truth_fp, index_column=index_column)
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

    print(
        f"[split->sequential] train_seq={len(train_seq)} val_seq={len(val_seq)} "
        f"test_seq={len(test_seq)} total={offset}"
    )
    return train_seq, val_seq, test_seq


# =====================================================
# LABEL: log10(energy/GeV)
# =====================================================

class Log10Energy(Label):
    def __init__(self, key: str = "energy_log10", energy_key: str = "energy"):
        super().__init__(key=key)
        self._energy_key = energy_key

    def __call__(self, graph) -> Tensor:
        e = graph[self._energy_key].reshape(-1).to(torch.float32)
        return torch.log10(e + 1e-24).reshape(1)


# =====================================================
# ROBUST SCALER
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
    for seq in chosen:
        g = dataset[seq]
        x = g.x.detach().cpu().numpy()
        if x.shape[0] > max_pulses_per_event:
            idx = rng.choice(x.shape[0], size=max_pulses_per_event, replace=False)
            x = x[idx]
        samples.append(x)

    X = np.concatenate(samples, axis=0)
    print(f"[robust-scaler] total_pulses_used={X.shape[0]} features={X.shape[1]}")

    med = np.median(X, axis=0)
    q25 = np.quantile(X, 0.25, axis=0)
    q75 = np.quantile(X, 0.75, axis=0)
    iqr = np.maximum(q75 - q25, eps)

    return RobustScalerParams(median=med.tolist(), iqr=iqr.tolist(), feature_names=list(feature_names))


def _save_scaler(params: RobustScalerParams, path: str) -> None:
    payload = {"feature_names": params.feature_names, "median": params.median, "iqr": params.iqr}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[robust-scaler] saved -> {path}")


def _load_scaler(path: str) -> RobustScalerParams:
    with open(path, "r") as f:
        payload = json.load(f)
    return RobustScalerParams(feature_names=payload["feature_names"], median=payload["median"], iqr=payload["iqr"])


# =====================================================
# DETECTOR-LIKE SCALER (GraphNeT compatible)
# =====================================================

class PONE:
    def __init__(
        self,
        *,
        feature_names: List[str],
        scaler: Optional[RobustScalerParams] = None,
        scaler_path: Optional[str] = None,
        scaler_params: Optional[RobustScalerParams] = None,
        string_index_name: str = "__unused__",
        eps: float = 1e-12,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> None:
        self.feature_names = list(feature_names)
        self.eps = float(eps)
        self.dtype = dtype
        self.string_index_name = string_index_name

        if scaler is None and scaler_params is not None:
            scaler = scaler_params
        if scaler is None and scaler_path is not None:
            scaler = _load_scaler(scaler_path)

        if scaler is None:
            self._median_np = np.zeros((len(self.feature_names),), dtype=np.float32)
            self._iqr_np = np.ones((len(self.feature_names),), dtype=np.float32)
            self._has_scaler = False
        else:
            assert scaler.feature_names == self.feature_names
            self._median_np = np.asarray(scaler.median, dtype=np.float32)
            self._iqr_np = np.maximum(np.asarray(scaler.iqr, dtype=np.float32), self.eps)
            self._has_scaler = True

        self._median_torch: dict[torch.device, Tensor] = {}
        self._iqr_torch: dict[torch.device, Tensor] = {}

    def has_scaler(self) -> bool:
        return bool(self._has_scaler)

    def set_scaler(self, scaler: RobustScalerParams) -> None:
        assert scaler.feature_names == self.feature_names
        self._median_np = np.asarray(scaler.median, dtype=np.float32)
        self._iqr_np = np.maximum(np.asarray(scaler.iqr, dtype=np.float32), self.eps)
        self._has_scaler = True
        self._median_torch.clear()
        self._iqr_torch.clear()

    def _get_torch_params(self, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if device not in self._median_torch:
            self._median_torch[device] = torch.tensor(self._median_np, device=device, dtype=dtype)
            self._iqr_torch[device] = torch.tensor(self._iqr_np, device=device, dtype=dtype)
        return self._median_torch[device], self._iqr_torch[device]

    # GraphNeT calls detector(x, input_feature_names)
    def __call__(self, x: Tensor, input_feature_names: Optional[List[str]] = None) -> Tensor:
        return self.transform_tensor(x)

    def transform_tensor(self, x: Tensor) -> Tensor:
        if x.numel() == 0:
            return x
        if x.dim() != 2:
            raise ValueError(f"Expected x [N,F], got {tuple(x.shape)}")
        if x.size(1) != len(self.feature_names):
            raise ValueError(f"Feature dim mismatch: x has {x.size(1)}, expected {len(self.feature_names)}")
        med, iqr = self._get_torch_params(device=x.device, dtype=x.dtype)
        return (x - med) / iqr


# =====================================================
# LIGHTNING MODULE + save_config()
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

    def save_config(self, path: str) -> None:
        # GraphnetEarlyStopping expects this method
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("model:\n")
            f.write(f"  class: {self.__class__.__name__}\n")
            f.write(f"  backbone: {self.backbone.__class__.__name__}\n")
            f.write(f"  task: {self.task.__class__.__name__}\n")
            f.write("hparams:\n")
            for k, v in dict(self.hparams).items():
                f.write(f"  {k}: {str(v).replace(chr(10), ' ')}\n")

    def forward(self, batch):
        z = self.backbone(batch)
        return self.task(z)

    def _shared_step(self, batch, stage: str) -> Tensor:
        pred = self.forward(batch)
        loss = self.task.compute_loss(pred, batch)

        y = batch["energy_log10"].reshape(-1, 1).to(pred.dtype)
        mae_log10 = torch.mean(torch.abs(pred - y))

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=BATCH_SIZE,
        )
        self.log(
            f"{stage}_mae_log10",
            mae_log10,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=BATCH_SIZE,
        )
        return loss

    def training_step(self, batch, batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")
    
    def save_state_dict(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)


    def validation_step(self, batch, batch_idx: int) -> None:
        _ = self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> None:
        _ = self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self._base_lr)

        # ---- IMPORTANT: guard milestones ----
        # steps_per_epoch can be 1 in your current run (596 events, batch_size=1024).
        total_steps = max(1, int(self._steps_per_epoch * self._max_epochs))
        warmup_steps = int(self._steps_per_epoch * self._warmup_fraction_first_epoch)
        warmup_steps = max(1, warmup_steps)
        warmup_steps = min(warmup_steps, total_steps - 1) if total_steps > 1 else 1

        f0 = self._min_lr / self._base_lr
        f1 = 1.0
        f2 = self._min_lr / self._base_lr

        sched = PiecewiseLinearLR(
            optimizer=opt,
            milestones=[0, warmup_steps, total_steps],
            factors=[f0, f1, f2],
            last_epoch=-1,
        )

        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}


# =====================================================
# BUILD DATASET + GRAPHDEF
# =====================================================

def _build_graph_definition(detector: PONE) -> GraphDefinition:
    node_def = NodesAsPulses(input_feature_names=FEATURES)
    edge_def = KNNEdges(nb_nearest_neighbours=8, columns=[0, 1, 2])

    return GraphDefinition(
        detector=detector,
        node_definition=node_def,
        edge_definition=edge_def,
        input_feature_names=FEATURES,
        dtype=torch.float32,
    )


def _build_base_dataset(detector: PONE) -> ParquetDataset:
    graph_def = _build_graph_definition(detector)

    ds = ParquetDataset(
        path=PARQUET_ROOT,
        pulsemaps=[PULSEMAP_TABLE],
        features=FEATURES,
        truth=TRUTH_COLUMNS,
        data_representation=graph_def,
        graph_definition=None,
        node_truth=None,
        index_column=INDEX_COLUMN,
        truth_table=TRUTH_TABLE,
        node_truth_table=None,
        string_selection=None,
        selection=None,
        dtype=torch.float32,
        cache_size=1,
        labels=None,
    )

    ds.add_label(key="energy_log10", fn=Log10Energy(key="energy_log10", energy_key="energy"))
    return ds


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    _ensure_dir(OUTPUT_DIR)

    # If you later enable workers, set spawn explicitly (safer on clusters).
    if NUM_WORKERS > 0:
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
        except Exception:
            pass

    print("============================================================")
    print("[setup] Discovering parquet chunks...")
    truth_files = _discover_chunk_files(PARQUET_ROOT, TRUTH_TABLE)
    feat_files = _discover_chunk_files(PARQUET_ROOT, PULSEMAP_TABLE)
    print(f"[setup] truth_chunks={len(truth_files)} features_chunks={len(feat_files)}")

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
    print("[dataset] Build dataset (identity scaler first) to fit robust scaler on TRAIN...")
    detector = PONE(feature_names=FEATURES, scaler_params=None, string_index_name="string")
    dataset = _build_base_dataset(detector)

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
    print("[dataset] Re-build dataset WITH robust scaler active...")
    dataset = _build_base_dataset(detector)

    train_ds = Subset(dataset, train_seq)
    val_ds = Subset(dataset, val_seq)
    test_ds = Subset(dataset, test_seq)
    print(f"[dataset] train_len={len(train_ds)} val_len={len(val_ds)} test_len={len(test_ds)}")

    print("============================================================")
    print("[dataloader] Creating PyG dataloaders...")
    dl_kwargs = dict(
        batch_size=BATCH_SIZE,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        num_workers=NUM_WORKERS,
    )
    if NUM_WORKERS > 0:
        dl_kwargs["persistent_workers"] = PERSISTENT_WORKERS
        dl_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = PyGDataLoader(train_ds, shuffle=SHUFFLE_TRAIN, **dl_kwargs)
    val_loader = PyGDataLoader(val_ds, shuffle=SHUFFLE_VAL, **dl_kwargs)
    test_loader = PyGDataLoader(test_ds, shuffle=SHUFFLE_TEST, **dl_kwargs)

    steps_per_epoch = len(train_loader)
    print(f"[dataloader] steps_per_epoch={steps_per_epoch}")
    print(f"[dataloader] NUM_WORKERS={NUM_WORKERS} PIN_MEMORY={PIN_MEMORY} PERSISTENT_WORKERS={dl_kwargs.get('persistent_workers', False)}")

    if DEBUG_FETCH_ONE_BATCH:
        print("[debug] fetching 1 val batch to confirm dataloader works...")
        b = next(iter(val_loader))
        print("[debug] ok, got batch type=", type(b), "num_graphs=", getattr(b, "num_graphs", None))

    print("============================================================")
    print("[model] Building DynEdge backbone...")
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
    print("[task] Building task + LogCoshLoss...")
    loss_fn = LogCoshLoss()
    task = IdentityTask(
        nb_outputs=1,
        target_labels="energy_log10",
        hidden_size=backbone.nb_outputs,
        loss_function=loss_fn,
        prediction_labels=["energy_log10_pred"],
    )

    lit_model = DynEdgeEnergyLightningModule(
        backbone=backbone,
        task=task,
        steps_per_epoch=steps_per_epoch,
        max_epochs=MAX_EPOCHS,
        base_lr=BASE_LR,
        min_lr=MIN_LR,
        warmup_fraction_first_epoch=WARMUP_FRACTION_OF_FIRST_EPOCH,
    )

    print("============================================================")
    print("[trainer] Configuring callbacks...")
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

    # optional perf hint for tensor cores
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    trainer = Trainer(
        default_root_dir=OUTPUT_DIR,
        accelerator=accelerator,
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        log_every_n_steps=50,
        enable_checkpointing=False,
        deterministic=True,
        benchmark=False,
        num_sanity_val_steps=NUM_SANITY_VAL_STEPS,  # default 0 (prevents hang in sanity check)
    )

    print("============================================================")
    print("[train] Starting training...")
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("============================================================")
    print("[validate] Running validation...")
    val_metrics = trainer.validate(lit_model, dataloaders=val_loader, verbose=True)
    print(f"[validate] metrics={val_metrics}")

    print("============================================================")
    print("[test] Running test...")
    test_metrics = trainer.test(lit_model, dataloaders=test_loader, verbose=True)
    print(f"[test] metrics={test_metrics}")

    print("============================================================")
    print("[done] Outputs in:", OUTPUT_DIR)
    print(" - robust scaler:", SCALER_PATH)
    print(" - best model state_dict:", os.path.join(OUTPUT_DIR, "best_model.pth"))
    print("============================================================")


if __name__ == "__main__":
    main()

# Run:
#   python -u Makale.py 2>&1 | tee run.log
#
# If you want workers later (after it runs stable with 0):
#   export NUM_WORKERS=2
#   export PERSISTENT_WORKERS=1
#   export PIN_MEMORY=1
#   python -u Makale.py 2>&1 | tee run.log
