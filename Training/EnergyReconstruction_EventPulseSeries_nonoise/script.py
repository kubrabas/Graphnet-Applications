#!/usr/bin/env python3
# train_dynedge_energy_pipeline.py

import os
import sys
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import pandas as pd
import pytorch_lightning as pl
from torch_geometric.data import Data

# -----------------------
# 0) Repo import
# -----------------------
# Go three levels up: .../Graphnet-Applications )
repo_root = os.path.abspath("../../..")
sys.path.append(repo_root)

print(f"[Init] repo_root = {repo_root}")
print(f"[Init] repo_root exists? {os.path.exists(repo_root)}")

from MyClasses.detector_pone import PONE  # noqa: E402

# -----------------------
# 1) GraphNeT imports
# -----------------------
from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset  # noqa: E402
from graphnet.data.dataloader import DataLoader  # noqa: E402

from graphnet.models.graphs import KNNGraph  # noqa: E402
from graphnet.models.graphs.nodes import NodesAsPulses  # noqa: E402

from graphnet.models.gnn import DynEdge  # noqa: E402
from graphnet.models.standard_model import StandardModel  # noqa: E402

from graphnet.models.task import StandardLearnedTask  # noqa: E402
from graphnet.training.loss_functions import LogCoshLoss  # noqa: E402

from graphnet.training.callbacks import (  # noqa: E402
    PiecewiseLinearLR,
    GraphnetEarlyStopping,
    ProgressBar,
)

from graphnet.utilities.maths import eps_like  # noqa: E402
from torch import Tensor  # noqa: E402


# -----------------------
# 2) Config
# -----------------------
@dataclass
class Cfg:
    # Data paths
    train_path: str = "/project/def-nahee/kbas/POM_Response_Parquet/merged/train_reindexed"
    val_path: str = "/project/def-nahee/kbas/POM_Response_Parquet/merged/val_reindexed"
    test_path: str = "/project/def-nahee/kbas/POM_Response_Parquet/merged/test_reindexed"

    # ParquetDataset keys
    pulsemaps: str = "features"
    truth_table: str = "truth"

    # Features / truth
    features: Tuple[str, ...] = ("pmt_x", "pmt_y", "pmt_z", "dom_time", "charge")
    truth: Tuple[str, ...] = ("energy",)

    # Loader
    batch_size: int = 1024
    num_workers: int = 8
    multiprocessing_context: str = "spawn"
    persistent_workers: bool = True
    pin_memory: bool = True

    # Training (paper)
    max_epochs: int = 30
    base_lr: float = 1e-5
    peak_lr: float = 1e-3
    early_stopping_patience: int = 5

    # If 1024 doesn't fit: micro-batch + grad accumulation => effective batch ~1024
    accumulate_grad_batches: int = 1

    # Model
    nb_neighbours: int = 8
    global_pooling_schemes: Tuple[str, ...] = ("min", "max", "mean", "sum")
    add_global_variables_after_pooling: bool = True
    add_norm_layer: bool = False
    skip_readout: bool = False

    # energy transform support (for inverse-transform validation)
    transform_support: Tuple[float, float] = (1e1, 1e8)

    # Outputs
    save_dir: str = "./runs/dynedge_energy"
    test_csv_name: str = "test_predictions.csv"


# -----------------------
# 3) log10 <-> energy transforms
# -----------------------
def logarithm(E: torch.Tensor) -> torch.Tensor:
    E_safe = torch.clamp(E, min=eps_like(E))
    return torch.log10(E_safe)


def exponential(t: torch.Tensor) -> torch.Tensor:
    return torch.pow(10.0, t)


# -----------------------
# 4) Task head
# -----------------------
class DepositedEnergyLog10Task(StandardLearnedTask):
    default_target_labels = ["energy"]
    default_prediction_labels = ["log10_energy_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        return x[:, :1]


# -----------------------
# 5) Helpers
# -----------------------
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


# -----------------------
# 6) Build datasets & loaders
# -----------------------
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


# -----------------------
# 7) Build model
# -----------------------
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

    print("[Model] Building energy task (log10 target + LogCosh)")
    task = DepositedEnergyLog10Task(
        hidden_size=backbone.nb_outputs,
        loss_function=LogCoshLoss(),
        target_labels=["energy"],
        prediction_labels=["log10_energy_pred"],
        transform_target=logarithm,
        transform_inference=exponential,
        transform_support=cfg.transform_support,
        loss_weight=None,
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


# -----------------------
# 8) Train + test
# -----------------------
def run(cfg: Cfg):
    pl.seed_everything(123, workers=True)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("\n========== CONFIG ==========")
    for k, v in cfg.__dict__.items():
        print(f"{k}: {v}")
    print("============================\n")

    data_representation, train_loader, val_loader, test_loader = build_data(cfg)

    # Sanity check loader behavior
    b0 = next(iter(train_loader))
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

    print("\n[Train] Starting Trainer.fit()")
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[ProgressBar(), early_stop],
        enable_checkpointing=False,
        log_every_n_steps=50,
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
                pred_log10 = preds_list[0].detach().float().squeeze(-1).cpu()

                true_E = extract_field(batch, "energy").detach().float().squeeze(-1).cpu()

                pred_E = exponential(pred_log10)
                true_log10 = logarithm(true_E)
                residual_log10 = (pred_log10 - true_log10)

                event_id = maybe_extract_event_id(batch)
                if event_id is not None:
                    event_id = event_id.detach().cpu().squeeze(-1)

                if ib % 20 == 0:
                    print(f"[Test] batch {ib}/{len(test_loader)}: N={len(pred_log10)} "
                          f"pred_log10 mean={pred_log10.mean().item():.3f} "
                          f"resid mean={residual_log10.mean().item():.3f}")

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
        out_csv = os.path.join(cfg.save_dir, cfg.test_csv_name)
        df.to_csv(out_csv, index=False)
        print(f"\n[Test] Wrote CSV: {out_csv}")
        print(f"[Test] Rows: {len(df)}")

        r = df["residual_log10"].to_numpy()
        p16, p50, p84 = (float(pd.Series(r).quantile(q)) for q in [0.16, 0.50, 0.84])
        W = (p84 - p16) / 2.0
        print(f"[Test] residual_log10 quantiles: p16={p16:.4f}, p50={p50:.4f}, p84={p84:.4f}, W={(W):.4f}")

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

    
    
## read this script from scratch and check if everything is correct