# =======================
# 1) Imports + log suppression
# =======================

import logging
import copy
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import time
from datetime import datetime

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch_geometric.data import Data

# ---- GraphNeT optional-deps warning suppress ----
class _SuppressGraphnetOptionalDepsAtImport(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith("graphnet"):
            return True
        msg = record.getMessage()
        if (
            "has_jammy_flows_package" in msg
            or "jammy_flows" in msg
            or "has_icecube_package" in msg
            or "`icecube` not available" in msg
            or "has_km3net_package" in msg
            or "`km3net` not available" in msg
        ):
            return False
        return True

if not getattr(logging, "_GRAPHNET_OPTIONAL_DEPS_FILTER_INSTALLED", False):
    _f_import = _SuppressGraphnetOptionalDepsAtImport()
    logging.getLogger("graphnet").addFilter(_f_import)
    logging.getLogger("graphnet.utilities.imports").addFilter(_f_import)
    logging._GRAPHNET_OPTIONAL_DEPS_FILTER_INSTALLED = True





from graphnet.training.callbacks import PiecewiseLinearLR, GraphnetEarlyStopping
from graphnet.training.loss_functions import LogCoshLoss, VonMisesFisher2DLoss


# =======================
# 2) GraphNeT imports
# =======================

from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset
from graphnet.data.dataloader import DataLoader
from graphnet.models.data_representation import KNNGraph, NodesAsPulses
from graphnet.models.detector.detector import Detector
from graphnet.models.gnn import DynEdge
from graphnet.models.standard_model import StandardModel
from graphnet.models.task.reconstruction import (
    AzimuthReconstructionWithKappa,
    ZenithReconstructionWithKappa,
)
from graphnet.models.task import StandardLearnedTask
from graphnet.utilities.maths import eps_like


# =======================
# 3) Logging helpers (epoch-in-earlystopping logs)
# =======================

_EPOCH_CTX = {"epoch": None}

class _EpochContextCallback(Callback):
    def _set_epoch(self, trainer):
        if trainer.sanity_checking:
            return
        _EPOCH_CTX["epoch"] = trainer.current_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        self._set_epoch(trainer)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._set_epoch(trainer)

class _InjectEpochIntoEarlyStoppingLog(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        ep = _EPOCH_CTX.get("epoch", None)
        if ep is not None and msg.startswith("Metric ") and "New best score" in msg:
            record.msg = msg + f" | epoch: {ep}"
            record.args = ()
        return True

_LOG_FILTERS_INSTALLED = False

def install_logging_filters() -> None:
    global _LOG_FILTERS_INSTALLED
    if _LOG_FILTERS_INSTALLED:
        return
    _LOG_FILTERS_INSTALLED = True

    es_filter = _InjectEpochIntoEarlyStoppingLog()
    logging.getLogger("pytorch_lightning.callbacks.early_stopping").addFilter(es_filter)
    logging.getLogger("lightning.pytorch.callbacks.early_stopping").addFilter(es_filter)
    logging.getLogger("graphnet.training.callbacks").addFilter(es_filter)


# =======================
# 4) Detector: PONE robust scaling (paper Eq. 2.1)
# =======================

class PONE(Detector):
    """Detector class for P-ONE."""

    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"

    def __init__(
        self,
        replace_with_identity: Optional[List[str]] = None,
        percentiles_csv: str = "/project/def-nahee/kbas/Graphnet-Applications/MyClasses/train_feature_percentiles_p25_p50_p75.csv",
        eps: float = 1e-12,
        selected_features: Optional[List[str]] = None,  
    ) -> None:
        super().__init__(replace_with_identity=replace_with_identity)

        df = pd.read_csv(percentiles_csv)
        p = df.set_index("feature")
        self._p25 = p["p25"].to_dict()
        self._p50 = p["p50"].to_dict()
        self._p75 = p["p75"].to_dict()
        self._eps = eps

        self._selected_features = selected_features      

    def _robust_scale(self, x: torch.Tensor, feature: str) -> torch.Tensor:
        if feature not in self._p50:
            raise KeyError(f"Percentiles not found for feature='{feature}'")

        p25 = float(self._p25[feature])
        p50 = float(self._p50[feature])
        p75 = float(self._p75[feature])

        denom = (p75 - p25)
        x = x.to(torch.float32)

        if abs(denom) < self._eps:
            return x - p50
        return (x - p50) / denom

    def feature_map(self) -> Dict[str, Callable]:
        full = {
            "charge": self._charge,
            "dom_time": self._dom_time,
            "dom_x": self._dom_x,
            "dom_y": self._dom_y,
            "dom_z": self._dom_z,
            "string": self._string,
            "pmt_number": self._pmt_number,
            "dom_number": self._dom_number,
            "pmt_x": self._pmt_x,
            "pmt_y": self._pmt_y,
            "pmt_z": self._pmt_z,
        }

        if self._selected_features is None:
            return full

        return {k: full[k] for k in self._selected_features}

    def _charge(self, x: torch.Tensor) -> torch.Tensor:
        return self._robust_scale(x, "charge")

    def _dom_time(self, x: torch.Tensor) -> torch.Tensor:
        return self._robust_scale(x, "dom_time")

    def _dom_x(self, x: torch.Tensor) -> torch.Tensor:
        return self._robust_scale(x, "dom_x")

    def _dom_y(self, x: torch.Tensor) -> torch.Tensor:
        return self._robust_scale(x, "dom_y")

    def _dom_z(self, x: torch.Tensor) -> torch.Tensor:
        return self._robust_scale(x, "dom_z")

    def _string(self, x: torch.Tensor) -> torch.Tensor:
        return self._robust_scale(x, "string")

    def _pmt_number(self, x: torch.Tensor) -> torch.Tensor:
        return self._robust_scale(x, "pmt_number")

    def _dom_number(self, x: torch.Tensor) -> torch.Tensor:
        return self._robust_scale(x, "dom_number")

    def _pmt_x(self, x: torch.Tensor) -> torch.Tensor:
        return self._robust_scale(x, "pmt_x")

    def _pmt_y(self, x: torch.Tensor) -> torch.Tensor:
        return self._robust_scale(x, "pmt_y")

    def _pmt_z(self, x: torch.Tensor) -> torch.Tensor:
        return self._robust_scale(x, "pmt_z")

# =======================
# 5) Callbacks: metrics.csv, epoch_time.csv
# =======================


class EpochTimeLogger(Callback):
    """
    Writes epoch start timestamps (minutes since fit start) to CSV.
    Also prints per-epoch duration to stdout (.out) using deltas between starts.
    """
    def __init__(self, out_dir, filename: str = "epoch_time.csv"):
        self.out_dir = Path(out_dir)
        self.file = self.out_dir / filename
        self.t0 = None
        self.prev_elapsed_min = 0.0

    def on_fit_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self.t0 = time.time()
        self.prev_elapsed_min = 0.0

        self.out_dir.mkdir(parents=True, exist_ok=True)
        with self.file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "time_min"])
            writer.writeheader()
            writer.writerow({"epoch": 0, "time_min": "0.000"})  # epoch 0 starts at t=0

        print(f"[Time] fit start {datetime.now().isoformat(timespec='seconds')} | {self.file}")

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking or self.t0 is None:
            return

        ep = trainer.current_epoch
        elapsed_min = (time.time() - self.t0) / 60.0

        # For ep>0, we can report duration of previous epoch (train+val+overheads)
        if ep > 0:
            prev_dur = elapsed_min - self.prev_elapsed_min
            print(
                f"[Time] epoch {ep-1} duration = {prev_dur:.2f} min | "
                f"elapsed = {elapsed_min:.2f} min | {datetime.now().isoformat(timespec='seconds')}"
            )

            # write this epoch's start time
            with self.file.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "time_min"])
                writer.writerow({"epoch": ep, "time_min": f"{elapsed_min:.3f}"})

        self.prev_elapsed_min = elapsed_min

    def on_fit_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.t0 is None:
            return

        elapsed_min = (time.time() - self.t0) / 60.0
        # last_ep = trainer.current_epoch  # last completed epoch index
        last_ep = max(0, trainer.current_epoch - 1)
        last_dur = elapsed_min - self.prev_elapsed_min

        print(
            f"[Time] epoch {last_ep} duration = {last_dur:.2f} min | "
            f"total = {elapsed_min:.2f} min | {datetime.now().isoformat(timespec='seconds')}"
        )



class EpochCSVLogger(Callback):
    def __init__(self, out_dir, filename: str = "metrics.csv"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.file = self.out_dir / filename

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        row = {
            "epoch": trainer.current_epoch,
            "train_loss": metrics.get("train_loss_epoch", metrics.get("train_loss")),
            "val_loss": metrics.get("val_loss"),
        }
        for k, v in row.items():
            if torch.is_tensor(v):
                row[k] = v.item()

        write_header = not self.file.exists()
        with self.file.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# =======================
# 6) Energy task (log10 target + LogCosh)
# =======================

def logarithm(E: torch.Tensor) -> torch.Tensor:
    E_safe = torch.clamp(E, min=eps_like(E))
    return torch.log10(E_safe)

def exponential(t: torch.Tensor) -> torch.Tensor:
    return torch.pow(10.0, t)

class DepositedEnergyLog10Task(StandardLearnedTask):
    default_target_labels = ["energy"]
    default_prediction_labels = ["log10_energy_pred"]
    nb_inputs = 1

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


# =======================
# 7) Helpers
# =======================

def extract_field(batch, field: str) -> torch.Tensor:
    if isinstance(batch, Data):
        return batch[field]
    if isinstance(batch, (list, tuple)):
        return torch.cat([b[field] for b in batch], dim=0)
    raise TypeError(f"Unsupported batch type: {type(batch)}")

def move_batch_to_device(batch, device):
    if isinstance(batch, Data):
        return batch.to(device)
    return [b.to(device) for b in batch]

def maybe_extract_event_id(batch) -> Optional[torch.Tensor]:
    for key in ["event_id", "event_no", "event", "idx"]:
        try:
            return extract_field(batch, key)
        except Exception:
            pass
    return None

def _circular_abs_diff(a: torch.Tensor, b: torch.Tensor, period: float = 2 * math.pi) -> torch.Tensor:
    d = torch.remainder(a - b + period / 2, period) - period / 2
    return d.abs()


# =======================
# 8) Config (paper-like)
# =======================

@dataclass
class Cfg:
    seed: int = 20260202

    train_path: str = "/project/def-nahee/kbas/POM_Response_Parquet/merged/train_reindexed"
    val_path: str = "/project/def-nahee/kbas/POM_Response_Parquet/merged/val_reindexed"
    test_path: str = "/project/def-nahee/kbas/POM_Response_Parquet/merged/test_reindexed"

    pulsemaps: str = "features"
    truth_table: str = "truth"

    # Node features you use
    features: Tuple[str, ...] = ("pmt_x", "pmt_y", "pmt_z", "dom_time", "charge")

    # Build dataset once with all needed truths; each task will pick what it needs.
    truth_all: Tuple[str, ...] = ("azimuth", "zenith", "energy")

    batch_size: int = 256
    num_workers: int = 8
    multiprocessing_context: str = "spawn"
    persistent_workers: bool = True
    pin_memory: bool = True

    # Paper: 30 epoch budget, patience=5
    max_epochs: int = 30
    early_stopping_patience: int = 5

    # Paper LR schedule endpoints
    base_lr: float = 1e-5
    peak_lr: float = 1e-3

    # Effective batch 1024 = 256 * 4
    accumulate_grad_batches: int = 4

    nb_neighbours: int = 8
    global_pooling_schemes: Tuple[str, ...] = ("min", "max", "mean", "sum")
    add_global_variables_after_pooling: bool = True
    add_norm_layer: bool = False
    skip_readout: bool = False

    # For energy inverse-transform stability
    transform_support: Tuple[float, float] = (1e1, 1e8)

    # Outputs
    save_dir: str = "/project/6061446/kbas/Graphnet-Applications/Training/Combined_EventPulseSeries_nonoise/initiative7"
    metrics_name: str = "metrics.csv"
    test_csv_name: str = "test_predictions.csv"


# =======================
# 9) Build data once (shared across tasks)
# =======================

def build_data(cfg: Cfg):
    print("[Data] Building data_representation (KNNGraph + robust scaling)")
    data_representation = KNNGraph(
        detector=PONE(selected_features=list(cfg.features)),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=cfg.nb_neighbours,
        distance_as_edge_feature=False,  # paper doesn't describe using distance as edge feature
    )

    print("[Data] Creating ParquetDataset(s)")
    train_ds = ParquetDataset(
        path=cfg.train_path,
        pulsemaps=cfg.pulsemaps,
        truth_table=cfg.truth_table,
        features=list(cfg.features),
        truth=list(cfg.truth_all),
        data_representation=data_representation,
    )

    val_ds = ParquetDataset(
        path=cfg.val_path,
        pulsemaps=cfg.pulsemaps,
        truth_table=cfg.truth_table,
        features=list(cfg.features),
        truth=list(cfg.truth_all),
        data_representation=data_representation,
    )

    test_ds = None
    if cfg.test_path:
        test_ds = ParquetDataset(
            path=cfg.test_path,
            pulsemaps=cfg.pulsemaps,
            truth_table=cfg.truth_table,
            features=list(cfg.features),
            truth=list(cfg.truth_all),
            data_representation=data_representation,
        )

    print("[Data] Creating DataLoader(s)")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        multiprocessing_context=cfg.multiprocessing_context,
        persistent_workers=cfg.persistent_workers,
        pin_memory=cfg.pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        multiprocessing_context=cfg.multiprocessing_context,
        persistent_workers=cfg.persistent_workers,
        pin_memory=cfg.pin_memory,
    )

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            multiprocessing_context=cfg.multiprocessing_context,
            persistent_workers=cfg.persistent_workers,
            pin_memory=cfg.pin_memory,
        )

    print(f"[Data] len(train_loader) = {len(train_loader)} batches/epoch (drop_last=True)")
    print(f"[Data] len(val_loader)   = {len(val_loader)} batches/epoch")
    if test_loader is not None:
        print(f"[Data] len(test_loader)  = {len(test_loader)} batches")

    return data_representation, train_loader, val_loader, test_loader


# =======================
# 10) Build model per target
# =======================

def build_model(cfg: Cfg, data_representation, steps_per_epoch_optimizer: int, target: str):
    backbone = DynEdge(
        nb_inputs=len(cfg.features),
        nb_neighbours=cfg.nb_neighbours,
        global_pooling_schemes=list(cfg.global_pooling_schemes),
        add_global_variables_after_pooling=cfg.add_global_variables_after_pooling,
        add_norm_layer=cfg.add_norm_layer,
        skip_readout=cfg.skip_readout,
    )

    if target == "zenith":
        task = ZenithReconstructionWithKappa(
            hidden_size=backbone.nb_outputs,
            loss_function=VonMisesFisher2DLoss(),
            target_labels=["zenith"],
        )
    elif target == "azimuth":
        task = AzimuthReconstructionWithKappa(
            hidden_size=backbone.nb_outputs,
            loss_function=VonMisesFisher2DLoss(),
            target_labels=["azimuth"],
        )
    elif target == "energy":
        task = DepositedEnergyLog10Task(
            hidden_size=backbone.nb_outputs,
            loss_function=LogCoshLoss(),
            target_labels=["energy"],
            prediction_labels=["log10_energy_pred"],
            transform_target=lambda E: torch.log10(torch.clamp(E, min=eps_like(E))),
            transform_inference=lambda t: torch.pow(10.0, t),
            transform_support=cfg.transform_support,
            loss_weight=None,
        )
    else:
        raise ValueError(f"Unknown target={target}")

    # Paper one-cycle-like schedule (step-based)
    total_steps = steps_per_epoch_optimizer * cfg.max_epochs
    warmup_steps = max(1, int(0.5 * steps_per_epoch_optimizer))
    peak_factor = cfg.peak_lr / cfg.base_lr

    model = StandardModel(
        tasks=task,
        data_representation=data_representation,
        backbone=backbone,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": cfg.base_lr},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [0, warmup_steps, total_steps],
            "factors": [1.0, peak_factor, 1.0],
        },
        scheduler_config={"interval": "step"},
    )
    return model


# =======================
# 11) Test writers (single-write for speed)
# =======================

def run_test(cfg: Cfg, target: str, model: pl.LightningModule, test_loader, out_dir: str) -> None:
    if test_loader is None:
        print(f"[Test={target}] No test_loader. Skipping.")
        return

    best_path = os.path.join(out_dir, "best_model.pth")
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        print(f"[Test={target}] Loaded best weights: {best_path}")
    else:
        print(f"[Test={target}][WARN] best_model.pth not found, using last weights.")

    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    else:
        for p in model.parameters():
            p.requires_grad = False

    device = next(model.parameters()).device
    out_csv = os.path.join(out_dir, cfg.test_csv_name)

    rows = []
    with torch.no_grad():
        for ib, batch in enumerate(test_loader):
            batch = move_batch_to_device(batch, device)

            preds_list = model(batch)
            pred0 = preds_list[0].detach().float()

            event_id = maybe_extract_event_id(batch)
            if event_id is not None:
                event_id = event_id.detach().cpu().view(-1)

            if target in ["zenith", "azimuth"]:
                # pred shape [N,2] = [angle, kappa]
                if pred0.ndim != 2 or pred0.shape[1] != 2:
                    raise RuntimeError(f"[Test={target}] Expected pred [N,2], got {tuple(pred0.shape)}")

                pred_angle = pred0[:, 0].cpu()
                kappa = pred0[:, 1].cpu()
                truth = extract_field(batch, target).detach().float().view(-1).cpu()

                if target == "azimuth":
                    err = _circular_abs_diff(pred_angle, truth)
                else:
                    err = (pred_angle - truth).abs()

                err_deg = err * (180.0 / math.pi)

                for i in range(len(err_deg)):
                    row = {
                        f"true_{target}": float(truth[i].item()),
                        f"pred_{target}": float(pred_angle[i].item()),
                        "abs_error_deg": float(err_deg[i].item()),
                        "kappa": float(kappa[i].item()),
                    }
                    if event_id is not None and i < len(event_id):
                        row["event_id"] = int(event_id[i].item())
                    rows.append(row)

                if ib % 50 == 0:
                    print(f"[Test={target}] batch {ib}/{len(test_loader)} | median_abs_error_deg={err_deg.median().item():.3f}")

            else:
                # energy: pred is log10(E)
                pred_log10 = pred0.squeeze(-1).cpu()
                true_E = extract_field(batch, "energy").detach().float().view(-1).cpu()

                pred_E = exponential(pred_log10)
                true_log10 = logarithm(true_E)
                residual_log10 = (pred_log10 - true_log10)

                if ib % 20 == 0:
                    print(f"[Test=energy] batch {ib}/{len(test_loader)} | resid mean={residual_log10.mean().item():.3f}")

                for i in range(len(pred_log10)):
                    row = {
                        "true_energy": float(true_E[i].item()),
                        "pred_log10_energy": float(pred_log10[i].item()),
                        "pred_energy": float(pred_E[i].item()),
                        "residual_log10": float(residual_log10[i].item()),
                    }
                    if event_id is not None and i < len(event_id):
                        row["event_id"] = int(event_id[i].item())
                    rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[Test={target}] Wrote: {out_csv} | rows={len(df)}")

    # Quick summary
    if target in ["zenith", "azimuth"]:
        a = torch.tensor(df["abs_error_deg"].to_numpy(), dtype=torch.float32)
        p16 = torch.quantile(a, 0.16).item()
        p50 = torch.quantile(a, 0.50).item()
        p84 = torch.quantile(a, 0.84).item()
        kmean = float(df["kappa"].mean())
        print(f"[Test={target}] abs_error_deg: p16={p16:.3f}, p50={p50:.3f}, p84={p84:.3f} | kappa_mean={kmean:.3f}")
    else:
        r = df["residual_log10"].to_numpy()
        p16, p50, p84 = (float(pd.Series(r).quantile(q)) for q in [0.16, 0.50, 0.84])
        W = (p84 - p16) / 2.0
        print(f"[Test=energy] residual_log10: p16={p16:.4f}, p50={p50:.4f}, p84={p84:.4f}, W={W:.4f}")


# =======================
# 12) Run one target
# =======================

def run_one(cfg: Cfg, target: str, data_representation, train_loader, val_loader, test_loader):
    install_logging_filters()

    out_dir = os.path.join(cfg.save_dir, target)
    os.makedirs(out_dir, exist_ok=True)

    pl.seed_everything(cfg.seed, workers=True)

    # optimizer-step schedule needs "optimizer steps per epoch"
    steps_per_epoch_batches = len(train_loader)
    steps_per_epoch_optimizer = math.ceil(steps_per_epoch_batches / cfg.accumulate_grad_batches)

    model = build_model(cfg, data_representation, steps_per_epoch_optimizer, target=target)

    early_stop = GraphnetEarlyStopping(
        save_dir=out_dir,
        monitor="val_loss",
        mode="min",
        patience=cfg.early_stopping_patience,
        check_on_train_epoch_end=False,
        verbose=True,
    )

    metrics_cb = EpochCSVLogger(out_dir, filename=cfg.metrics_name)
    epoch_ctx_cb = _EpochContextCallback()
    time_cb = EpochTimeLogger(out_dir, filename="epoch_time.csv")


    # --- GPU sanity print
    print(f"\n[Run={target}] torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Run={target}] GPU = {torch.cuda.get_device_name(0)}")

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[epoch_ctx_cb, early_stop, metrics_cb, time_cb],
        enable_checkpointing=False,
        enable_progress_bar=False,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"[Run={target}] best_model: {os.path.join(out_dir, 'best_model.pth')}")
    print(f"[Run={target}] config:     {os.path.join(out_dir, 'config.yml')}")
    print(f"[Run={target}] metrics:    {os.path.join(out_dir, cfg.metrics_name)}")

    run_test(cfg, target=target, model=model, test_loader=test_loader, out_dir=out_dir)


# =======================
# 13) Main: run zenith -> azimuth -> energy
# =======================

if __name__ == "__main__":
    cfg = Cfg()
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("\n========== CONFIG ==========")
    for k, v in cfg.__dict__.items():
        print(f"{k}: {v}")
    print("============================\n")

    data_representation, train_loader, val_loader, test_loader = build_data(cfg)

    # quick sanity ranges
    b0 = next(iter(train_loader))
    az = extract_field(b0, "azimuth").detach().cpu().view(-1)
    ze = extract_field(b0, "zenith").detach().cpu().view(-1)
    en = extract_field(b0, "energy").detach().cpu().view(-1)
    print(f"[Sanity] azimuth range: {az.min():.3f}..{az.max():.3f}")
    print(f"[Sanity] zenith  range: {ze.min():.3f}..{ze.max():.3f}")
    print(f"[Sanity] energy  range: {en.min():.3e}..{en.max():.3e}")

    for target in ["energy", "zenith", "azimuth"]:
        run_one(cfg, target, data_representation, train_loader, val_loader, test_loader)
