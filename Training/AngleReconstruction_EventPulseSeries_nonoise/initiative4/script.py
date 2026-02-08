
# =======================
# 0) Imports
# =======================

import logging


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
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.labels import Direction
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
# 3) Lightning callbacks (metrics CSV + opening angle)
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
            "val_loss": metrics.get("val_loss"),
            "val_opening_angle_p16_deg": metrics.get("val_opening_angle_p16_deg"),
            "val_opening_angle_median_deg": metrics.get("val_opening_angle_median_deg"),
            "val_opening_angle_p84_deg": metrics.get("val_opening_angle_p84_deg"),
            "val_kappa_mean": metrics.get("val_kappa_mean"),
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

class ValOpeningAngleLogger(Callback):
    def __init__(self):
        self._angles = []
        self._kappas = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self._angles.clear()
        self._kappas.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            batch = move_batch_to_device(batch, pl_module.device)
            preds_list = pl_module(batch)
            pred = preds_list[0].detach()
            
            pred_dir = pred[:, :3]
            kappa = pred[:, 3]

            pred_dir = pred_dir / pred_dir.norm(dim=1, keepdim=True).clamp_min(1e-12)

           
            true_dir = extract_field(batch, "direction").detach()
            true_dir = true_dir / true_dir.norm(dim=1, keepdim=True).clamp_min(1e-12)


            cosang = (pred_dir * true_dir).sum(dim=1).clamp(-1.0, 1.0)
            ang_deg = torch.acos(cosang) * (180.0 / math.pi)

            self._angles.append(ang_deg.cpu())
            self._kappas.append(kappa.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if len(self._angles) == 0:
            return

        angles = torch.cat(self._angles).float()
        kappas = torch.cat(self._kappas).float()

        # quantiles
        p16 = torch.quantile(angles, 0.16)
        p50 = torch.quantile(angles, 0.50)
        p84 = torch.quantile(angles, 0.84)
        kappa_mean = kappas.mean()

        
        pl_module.log("val_opening_angle_p16_deg", p16, prog_bar=False, on_epoch=True, sync_dist=True)
        pl_module.log("val_opening_angle_median_deg", p50, prog_bar=False, on_epoch=True, sync_dist=True)
        pl_module.log("val_opening_angle_p84_deg", p84, prog_bar=False, on_epoch=True, sync_dist=True)
        pl_module.log("val_kappa_mean", kappa_mean, prog_bar=False, on_epoch=True, sync_dist=True)


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
    save_dir: str = "/project/6061446/kbas/Graphnet-Applications/Training/AngleReconstruction_EventPulseSeries_nonoise/initiative4"
    test_csv_name: str = "test_predictions.csv"  


    # Metrics (epoch-level)
    metrics_dir: str = "/project/6061446/kbas/Graphnet-Applications/Training/AngleReconstruction_EventPulseSeries_nonoise/initiative4"
    metrics_name: str = "metrics.csv"


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
    return torch.cat([d[field] for d in batch], dim=0)


def maybe_extract_event_id(batch) -> Optional[torch.Tensor]:
    for key in ["event_id", "event_no", "event", "idx"]:
        try:
            v = extract_field(batch, key)
            return v
        except Exception:
            pass
    return None


def move_batch_to_device(batch, device):
    if isinstance(batch, Data):
        return batch.to(device)
    return [d.to(device) for d in batch]


# =======================
# 6) Data (datasets + loaders)
# =======================


def build_data(cfg: Cfg):
    print("[Data] Building data_representation (KNNGraph + PONE robust scaling)")
    data_representation = KNNGraph(
        detector=PONE(selected_features=list(cfg.features)),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=cfg.nb_neighbours,
        distance_as_edge_feature=True,
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



    print("[Data] Adding derived label: direction (from azimuth, zenith)")
    direction_label = Direction()  # defaults: key="direction", azimuth_key="azimuth", zenith_key="zenith"

    train_ds.add_label(direction_label)
    val_ds.add_label(direction_label)
    if test_ds is not None:
        test_ds.add_label(direction_label)


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
# 7) Model (DynEdge + DirectionRecoWithKappa)
# =======================


def build_model(cfg: Cfg, data_representation, steps_per_epoch_optimizer: int):
    print("[Model] Building DynEdge backbone")
    backbone = DynEdge(
        nb_inputs=len(cfg.features),
        nb_neighbours=cfg.nb_neighbours,
        global_pooling_schemes=list(cfg.global_pooling_schemes),
        add_global_variables_after_pooling=cfg.add_global_variables_after_pooling,
        add_norm_layer=cfg.add_norm_layer,
        skip_readout=cfg.skip_readout,
    )

    print("[Model] Building direction task (vMF3D + kappa)")



    task = DirectionReconstructionWithKappa(
        hidden_size=backbone.nb_outputs,
        loss_function=VonMisesFisher3DLoss(),
        target_labels=["direction"],  
)

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
# 8) Train + Test (inference + CSV)
# =======================


def run(cfg: Cfg):
    install_logging_filters()
    pl.seed_everything(cfg.seed, workers=True)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("\n========== CONFIG ==========")
    for k, v in cfg.__dict__.items():
        print(f"{k}: {v}")
    print("============================\n")

    data_representation, train_loader, val_loader, test_loader = build_data(cfg)

    # Sanity check loader behavior
    b0 = next(iter(train_loader))

    az = extract_field(b0, "azimuth").detach().cpu().view(-1)
    ze = extract_field(b0, "zenith").detach().cpu().view(-1)

    d0 = extract_field(b0, "direction").detach().cpu()
    print(f"[Sanity] direction norm mean={d0.norm(dim=1).mean().item():.6f}") # expectation: 1.000000

    print(f"[Sanity] azimuth range in batch: {az.min():.3f} .. {az.max():.3f} | zenith: {ze.min():.3f} .. {ze.max():.3f}")

    print(f"[Sanity] batch type: {type(b0)}")
    print(f"[Sanity] graphs in one train batch: {num_graphs_in_batch(b0)} (expected {cfg.batch_size})")
    if num_graphs_in_batch(b0) != cfg.batch_size:
        print("[Sanity][WARN] graphs != batch_size. Might be OK, but verify dataset returns single-event graphs.")

    steps_per_epoch_batches = len(train_loader)
    steps_per_epoch_optimizer = math.ceil(steps_per_epoch_batches / cfg.accumulate_grad_batches)
    print(f"[Train] steps/epoch (batches)   = {steps_per_epoch_batches}")
    print(f"[Train] accumulate_grad_batches = {cfg.accumulate_grad_batches}")
    print(f"[Train] steps/epoch (optimizer) = {steps_per_epoch_optimizer}")

    model = build_model(cfg, data_representation, steps_per_epoch_optimizer)

    # Early stopping
    early_stop = GraphnetEarlyStopping(
        save_dir=cfg.save_dir,
        monitor="val_loss",
        mode="min",
        patience=cfg.early_stopping_patience,
        check_on_train_epoch_end=False,
        verbose=True,
    )



    metrics_cb = EpochCSVLogger(cfg.metrics_dir, filename=cfg.metrics_name)


    val_angle_cb = ValOpeningAngleLogger()


    epoch_ctx_cb = _EpochContextCallback()



    print("\n[Train] Starting Trainer.fit()")
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=1,
        # callbacks=[ProgressBar(), early_stop],
        # callbacks=[early_stop, val_angle_cb, metrics_cb],
        callbacks=[epoch_ctx_cb, early_stop, val_angle_cb, metrics_cb],
        enable_checkpointing=False,
        enable_progress_bar=False, 
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = os.path.join(cfg.save_dir, "best_model.pth")
    cfg_path = os.path.join(cfg.save_dir, "config.yml")
    print("\n[Train] Finished.")
    print(f"[Train] Best model path: {best_path}")
    print(f"[Train] Config path:     {cfg_path}")

    # Test inference + CSV
    if test_loader is not None:
        print("\n[Test] Starting inference on test set")
        model.eval()
        if hasattr(model, "freeze"):
            model.freeze()
        else:
            for p in model.parameters():
                p.requires_grad = False

        device = model.device
        print(f"[Test] model device = {device}")
        rows = []

        with torch.no_grad():
            for ib, batch in enumerate(test_loader):
                batch = move_batch_to_device(batch, device)

                preds_list = model(batch)

                pred = preds_list[0].detach().float().cpu()  # [N,4] = dir_x,dir_y,dir_z,kappa
                pred_dir = pred[:, :3]
                kappa = pred[:, 3]

                # normalize pred direction
                pred_dir = pred_dir / pred_dir.norm(dim=1, keepdim=True).clamp_min(1e-12)

                # true direction (dataset'e Direction label ekledik ya)
                true_dir = extract_field(batch, "direction").detach().float().cpu()  # [N,3]
                true_dir = true_dir / true_dir.norm(dim=1, keepdim=True).clamp_min(1e-12)


                # opening angle
                cosang = (pred_dir * true_dir).sum(dim=1).clamp(-1.0, 1.0)
                ang_deg = torch.acos(cosang) * (180.0 / math.pi)


                # true angles
                true_az = extract_field(batch, "azimuth").detach().float().cpu().view(-1)
                true_ze = extract_field(batch, "zenith").detach().float().cpu().view(-1)


                # pred angles (direction'dan geri çıkar)
                pred_ze = torch.acos(pred_dir[:, 2].clamp(-1.0, 1.0))
                pred_az = torch.atan2(pred_dir[:, 1], pred_dir[:, 0])
                pred_az = torch.where(pred_az < 0, pred_az + 2 * math.pi, pred_az)




                event_id = maybe_extract_event_id(batch)
                if event_id is not None:
                    event_id = event_id.detach().cpu().squeeze(-1)


                if ib % 20 == 0:
                    print(f"[Test] batch {ib}/{len(test_loader)}: N={len(ang_deg)} "
                          f"median_angle_deg={ang_deg.median().item():.3f} "
                          f"mean_kappa={kappa.mean().item():.3f}")    
                    

                
                for i in range(len(ang_deg)):
                     row = {
                         "opening_angle_deg": float(ang_deg[i].item()),
                         "kappa": float(kappa[i].item()),
                         "true_azimuth": float(true_az[i].item()),
                         "true_zenith": float(true_ze[i].item()),
                         "pred_azimuth": float(pred_az[i].item()),
                         "pred_zenith": float(pred_ze[i].item()),
                         }
                     if event_id is not None and i < len(event_id):
                         row["event_id"] = int(event_id[i].item())
                     rows.append(row)



        df = pd.DataFrame(rows)
        out_csv = os.path.join(cfg.save_dir, cfg.test_csv_name)
        df.to_csv(out_csv, index=False)
        print(f"\n[Test] Wrote CSV: {out_csv}")
        print(f"[Test] Rows: {len(df)}")

        
        a = df["opening_angle_deg"].to_numpy()
        p16, p50, p84 = (float(pd.Series(a).quantile(q)) for q in [0.16, 0.50, 0.84])
        W = (p84 - p16) / 2.0
        print(f"[Test] opening_angle_deg quantiles: p16={p16:.3f}, p50={p50:.3f}, p84={p84:.3f}, W={W:.3f}")


        # Small preview
        print("\n[Test] CSV head:")
        print(df.head(5).to_string(index=False))

    else:
        print("\n[Test] No test_loader (cfg.test_path is empty). Skipping.")


if __name__ == "__main__":
    cfg = Cfg()

    # If VRAM is not enough:
    # cfg.batch_size = 128
    # cfg.accumulate_grad_batches = 8  # effective batch ~1024

    run(cfg)

    
    