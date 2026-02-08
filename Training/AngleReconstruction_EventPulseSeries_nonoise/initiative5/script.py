## check after this script runs: is the files same as initiative3/4: config, best_model, metrics etc


# =======================
# 0) Imports
# =======================

import logging
import copy

# ---- GraphNeT optional-deps warning suppress  ----
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


import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch_geometric.data import Data

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
from graphnet.training.loss_functions import VonMisesFisher2DLoss
from graphnet.training.callbacks import PiecewiseLinearLR, GraphnetEarlyStopping


# =======================
# 1) Logging helpers (epoch-in-logs + suppress spam)
# =======================


# global epoch context 
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
        # EarlyStopping'in klasik mesajı:
        # "Metric val_loss improved by ... New best score: ..."
        msg = record.getMessage()
        ep = _EPOCH_CTX.get("epoch", None)
        if ep is not None and msg.startswith("Metric ") and "New best score" in msg:
            # msg zaten formatlanmış string -> record.msg'yi güncelle, args boşalt
            record.msg = msg + f" | epoch: {ep}"
            record.args = ()
        return True



_LOG_FILTERS_INSTALLED = False

# Install log filters once (avoid duplicate filters in notebook/re-runs)
def install_logging_filters() -> None:
    global _LOG_FILTERS_INSTALLED
    if _LOG_FILTERS_INSTALLED:
        return
    _LOG_FILTERS_INSTALLED = True

    # EarlyStopping log'una epoch ekle
    es_filter = _InjectEpochIntoEarlyStoppingLog()
    logging.getLogger("pytorch_lightning.callbacks.early_stopping").addFilter(es_filter)
    logging.getLogger("lightning.pytorch.callbacks.early_stopping").addFilter(es_filter)
    logging.getLogger("graphnet.training.callbacks").addFilter(es_filter)




# =======================
# 2) Detector: PONE (robust scaling)
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
# 3) Lightning callbacks (metrics CSV)
# =======================


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
            "val_loss": metrics.get("val_loss"),}


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
# 4) Config
# =======================



@dataclass
class Cfg:

    # seed
    seed: int = 20260202

    # Data paths
    train_path: str = "/project/def-nahee/kbas/POM_Response_Parquet/merged/train_reindexed"
    val_path: str = "/project/def-nahee/kbas/POM_Response_Parquet/merged/val_reindexed"
    test_path: str = "/project/def-nahee/kbas/POM_Response_Parquet/merged/test_reindexed"

    # ParquetDataset keys
    pulsemaps: str = "features"
    truth_table: str = "truth"

    # Features / truth
    features: Tuple[str, ...] = ("pmt_x", "pmt_y", "pmt_z", "dom_time", "charge")
    truth: Tuple[str, ...] =  ("azimuth", "zenith")

    # Loader
    batch_size: int = 128
    num_workers: int = 8
    multiprocessing_context: str = "spawn"
    persistent_workers: bool = True
    pin_memory: bool = True

    # Training (paper)
    max_epochs: int = 10
    base_lr: float = 1e-5
    peak_lr: float = 1e-3
    early_stopping_patience: int = 15

    # If 1024 doesn't fit: micro-batch + grad accumulation => effective batch ~1024
    accumulate_grad_batches: int = 8

    # Model
    nb_neighbours: int = 8
    global_pooling_schemes: Tuple[str, ...] = ("min", "max", "mean", "sum")
    add_global_variables_after_pooling: bool = True
    add_norm_layer: bool = False
    skip_readout: bool = False


    # Outputs
    save_dir: str = "/project/6061446/kbas/Graphnet-Applications/Training/AngleReconstruction_EventPulseSeries_nonoise/initiative5"


    # Metrics (epoch-level)
    metrics_name: str = "metrics.csv"

    # Test
    test_csv_name: str = "test_predictions.csv"



# =======================
# 5) Batch / tensor helpers
# =======================

def num_graphs_in_batch(batch) -> int:
    if isinstance(batch, Data):
        return int(batch.num_graphs) if hasattr(batch, "num_graphs") else 1
    return int(sum(getattr(d, "num_graphs", 1) for d in batch))


def extract_field(batch, field: str) -> torch.Tensor:
    if isinstance(batch, Data):
        return batch[field]
    if isinstance(batch, (list, tuple)):
        return torch.cat([b[field] for b in batch], dim=0)
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _wrap_2pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x, 2 * math.pi)


def _fold_0_pi(x: torch.Tensor) -> torch.Tensor:
    # map angle to [0, 2pi) then fold to [0, pi]
    x = _wrap_2pi(x)
    return torch.where(x > math.pi, 2 * math.pi - x, x)


def _circular_abs_diff(a: torch.Tensor, b: torch.Tensor, period: float = 2 * math.pi) -> torch.Tensor:
    # minimal absolute difference on a circle
    d = torch.remainder(a - b + period / 2, period) - period / 2
    return d.abs()


def _decode_angle_from_pred2d(mu2: torch.Tensor, target: str, swap_xy: bool) -> torch.Tensor:
    # mu2 shape [N,2] representing (cos,sin) OR (sin,cos) -> we will auto-pick by swap_xy
    x = mu2[:, 0]
    y = mu2[:, 1]
    if swap_xy:
        x, y = y, x

    ang = torch.atan2(y, x)
    ang = _wrap_2pi(ang)

    if target == "zenith":
        ang = _fold_0_pi(ang)

    return ang


def _choose_swap_xy(mu2: torch.Tensor, truth_angle: torch.Tensor, target: str) -> bool:
    # decide whether pred[:,0/1] is (cos,sin) or swapped, by picking smaller mean error
    ang0 = _decode_angle_from_pred2d(mu2, target=target, swap_xy=False)
    ang1 = _decode_angle_from_pred2d(mu2, target=target, swap_xy=True)

    if target == "azimuth":
        e0 = _circular_abs_diff(ang0, truth_angle).mean()
        e1 = _circular_abs_diff(ang1, truth_angle).mean()
    else:
        e0 = (ang0 - truth_angle).abs().mean()
        e1 = (ang1 - truth_angle).abs().mean()

    return bool((e1 < e0).item())

def move_batch_to_device(batch, device):
    if isinstance(batch, Data):
        return batch.to(device)
    return [b.to(device) for b in batch]


# =======================
# 6) Data (datasets + loaders)
# =======================


def build_data(cfg: Cfg):
    print("[Data] Building data_representation (KNNGraph + PONE robust scaling)")
    data_representation = KNNGraph(
        detector=PONE(selected_features=list(cfg.features)),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=cfg.nb_neighbours,
        distance_as_edge_feature=False,
    )

    print("[Data] Creating ParquetDataset(s)")
    train_ds = ParquetDataset(
        path=cfg.train_path,
        pulsemaps=cfg.pulsemaps,
        truth_table=cfg.truth_table,
        features=list(cfg.features),
        truth=list(cfg.truth),
        data_representation=data_representation,
    )

    val_ds = ParquetDataset(
        path=cfg.val_path,
        pulsemaps=cfg.pulsemaps,
        truth_table=cfg.truth_table,
        features=list(cfg.features),
        truth=list(cfg.truth),
        data_representation=data_representation,
    )

    test_ds = None
    if cfg.test_path:
        test_ds = ParquetDataset(
            path=cfg.test_path,
            pulsemaps=cfg.pulsemaps,
            truth_table=cfg.truth_table,
            features=list(cfg.features),
            truth=list(cfg.truth),
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
# 7) Model (DynEdge + {target} vMF2D + kappa)
# =======================


def build_model(cfg: Cfg, data_representation, steps_per_epoch_optimizer: int, target: str):
    print("[Model] Building DynEdge backbone")
    backbone = DynEdge(
        nb_inputs=len(cfg.features),
        nb_neighbours=cfg.nb_neighbours,
        global_pooling_schemes=list(cfg.global_pooling_schemes),
        add_global_variables_after_pooling=cfg.add_global_variables_after_pooling,
        add_norm_layer=cfg.add_norm_layer,
        skip_readout=cfg.skip_readout,
    )

    print(f"[Model] Building {target} task (vMF2D + kappa)")




    if target == "zenith":
        task = ZenithReconstructionWithKappa(
            hidden_size=backbone.nb_outputs,
            loss_function=VonMisesFisher2DLoss(),
            target_labels=["zenith"],    )

    elif target == "azimuth":
        task = AzimuthReconstructionWithKappa(
            hidden_size=backbone.nb_outputs,
            loss_function=VonMisesFisher2DLoss(),
            target_labels=["azimuth"],)
    else:
        raise ValueError(f"Unknown target={target}")


    # Paper LR schedule (implemented in optimizer-step units):
    total_steps = steps_per_epoch_optimizer * cfg.max_epochs
    warmup_steps = max(1, int(0.5 * steps_per_epoch_optimizer))
    peak_factor = cfg.peak_lr / cfg.base_lr

    print("[LR] Schedule setup:")
    print(f"     base_lr={cfg.base_lr:g}, peak_lr={cfg.peak_lr:g} (factor={peak_factor:g})")
    print(f"     steps/epoch (optimizer) = {steps_per_epoch_optimizer}")
    print(f"     warmup_steps            = {warmup_steps} (first 50% of epoch 0)")
    print(f"     total_steps             = {total_steps} (for {cfg.max_epochs} epochs)")

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
# 8) Train 
# =======================

def run_test(cfg: Cfg, target: str, model: pl.LightningModule, test_loader) -> None:
    if test_loader is None:
        print(f"[Test={target}] No test_loader. Skipping.")
        return

    best_path = os.path.join(cfg.save_dir, "best_model.pth")
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location="cpu")
        # bazen {"state_dict": ...} gelebiliyor; ikisini de kaldır
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

    out_csv = os.path.join(cfg.save_dir, cfg.test_csv_name)

    if os.path.exists(out_csv):
        os.remove(out_csv)
    write_header = True
    n_rows = 0


    swap_xy = None

    with torch.no_grad():
        for ib, batch in enumerate(test_loader):
            batch = move_batch_to_device(batch, device)

          

            preds = model(batch)
            pred = preds[0] if isinstance(preds, (list, tuple)) else preds
            pred = pred.detach()



           
            if pred.shape[1] < 3:
                raise RuntimeError(f"Expected pred shape [N,>=3], got {tuple(pred.shape)}")

            mu2 = pred[:, :2].float().cpu()
            kappa = pred[:, 2].float().cpu()

            truth = extract_field(batch, target).detach().float().cpu().view(-1)

            # pick swap once using first batch
            if swap_xy is None:
                swap_xy = _choose_swap_xy(mu2, truth, target=target)
                print(f"[Test={target}] decode: swap_xy={swap_xy}")

            pred_angle = _decode_angle_from_pred2d(mu2, target=target, swap_xy=swap_xy)

            if target == "azimuth":
                err = _circular_abs_diff(pred_angle, truth)  # radians
            else:
                err = (pred_angle - truth).abs()            # radians

            # write rows
            err_deg = err * (180.0 / math.pi)
            n_rows += len(err_deg)

            

            batch_rows = []
            for i in range(len(err_deg)):
                batch_rows.append(
                            {
                                f"true_{target}": float(truth[i].item()),
                                f"pred_{target}": float(pred_angle[i].item()),
                                "abs_error_deg": float(err_deg[i].item()),
                                "kappa": float(kappa[i].item()), }  )

            pd.DataFrame(batch_rows).to_csv(
                out_csv,
                mode="a",
                header=write_header,
                index=False,
            )
            write_header = False








            if ib % 50 == 0:
                print(f"[Test={target}] batch {ib}/{len(test_loader)} | median_abs_error_deg={err_deg.median().item():.3f}")

    
    print(f"[Test={target}] Wrote: {out_csv} | rows={n_rows}")
    df = pd.read_csv(out_csv, usecols=["abs_error_deg", "kappa"])



    a = torch.tensor(df["abs_error_deg"].to_numpy(), dtype=torch.float32)
    p16 = torch.quantile(a, 0.16).item()
    p50 = torch.quantile(a, 0.50).item()
    p84 = torch.quantile(a, 0.84).item()
    kmean = float(df["kappa"].mean())

    print(f"[Test={target}] abs_error_deg: p16={p16:.3f}, p50={p50:.3f}, p84={p84:.3f} | kappa_mean={kmean:.3f}")


def run_one(cfg: Cfg, target: str):
    install_logging_filters()
    cfg = copy.deepcopy(cfg)

    cfg.save_dir = os.path.join(cfg.save_dir, target)
    os.makedirs(cfg.save_dir, exist_ok=True)

    pl.seed_everything(cfg.seed, workers=True)

    data_representation, train_loader, val_loader, test_loader = build_data(cfg)

    # direction sanity check'i kaldır (şimdilik)
    b0 = next(iter(train_loader))
    az = extract_field(b0, "azimuth").detach().cpu().view(-1)
    ze = extract_field(b0, "zenith").detach().cpu().view(-1)
    print(f"[Sanity] azimuth range: {az.min():.3f}..{az.max():.3f} | zenith: {ze.min():.3f}..{ze.max():.3f}")

    steps_per_epoch_batches = len(train_loader)
    steps_per_epoch_optimizer = math.ceil(steps_per_epoch_batches / cfg.accumulate_grad_batches)

    model = build_model(cfg, data_representation, steps_per_epoch_optimizer, target=target)

    early_stop = GraphnetEarlyStopping(
        save_dir=cfg.save_dir,
        monitor="val_loss",
        mode="min",
        patience=cfg.early_stopping_patience,
        check_on_train_epoch_end=False,
        verbose=True,
    )

    metrics_cb = EpochCSVLogger(cfg.save_dir, filename=cfg.metrics_name)

    epoch_ctx_cb = _EpochContextCallback()

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[epoch_ctx_cb, early_stop, metrics_cb],  # ValOpeningAngleLogger'ı çıkar
        enable_checkpointing=False,
        enable_progress_bar=False,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"[Run={target}] best_model: {os.path.join(cfg.save_dir, 'best_model.pth')}")
    print(f"[Run={target}] config:     {os.path.join(cfg.save_dir, 'config.yml')}")
    print(f"[Run={target}] metrics:    {os.path.join(cfg.save_dir, cfg.metrics_name)}")
    run_test(cfg, target=target, model=model, test_loader=test_loader)


    return



if __name__ == "__main__":
    cfg = Cfg()
    for target in ["zenith", "azimuth"]:
        run_one(cfg, target)



