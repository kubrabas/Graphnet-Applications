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
import subprocess


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

def _read_rss_gb() -> float:
    """Return current process RSS memory in GB."""
    # Prefer /proc (Linux). Fallback to resource if needed.
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])  # kB
                    return kb / 1024.0 / 1024.0
    except Exception:
        pass

    try:
        import resource
        kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # On Linux, ru_maxrss is in KB
        return kb / 1024.0 / 1024.0
    except Exception:
        return float("nan")


def _read_system_mem_used_gb() -> float:
    """Return system-wide used memory (MemTotal - MemAvailable) in GB."""
    try:
        mem = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                key = line.split(":")[0]
                val = float(line.split(":")[1].strip().split()[0])  # kB
                mem[key] = val
        total = mem.get("MemTotal", float("nan"))
        avail = mem.get("MemAvailable", float("nan"))
        used = total - avail
        return used / 1024.0 / 1024.0
    except Exception:
        return float("nan")


def _cpu_load_pct() -> float:
    """
    Return an approximate CPU load percentage using loadavg(1min)/num_cpus.
    This is a rough proxy (not per-process CPU usage).
    """
    try:
        load1 = os.getloadavg()[0]
        ncpu = os.cpu_count() or 1
        return 100.0 * load1 / ncpu
    except Exception:
        return float("nan")


def _gpu_snapshot() -> Dict[str, float]:
    """
    Query GPU utilization and memory usage via nvidia-smi.

    Returns:
        dict with:
          - gpu_util_pct
          - gpu_mem_used_gb
          - gpu_mem_total_gb
          - gpu_mem_util_pct

    Notes:
      - If CUDA_VISIBLE_DEVICES is set, we target the first listed entry.
      - If nvidia-smi is unavailable, returns NaNs.
    """
    out = {
        "gpu_util_pct": float("nan"),
        "gpu_mem_used_gb": float("nan"),
        "gpu_mem_total_gb": float("nan"),
        "gpu_mem_util_pct": float("nan"),
    }

    try:
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_id = dev.split(",")[0].strip() if dev else None

        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        if gpu_id:
            cmd = ["nvidia-smi", "-i", gpu_id] + cmd[1:]

        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0 or not res.stdout.strip():
            return out

        line = res.stdout.strip().splitlines()[0]
        util_s, used_s, total_s = [x.strip() for x in line.split(",")]

        util = float(util_s)
        used_mb = float(used_s)
        total_mb = float(total_s)

        out["gpu_util_pct"] = util
        out["gpu_mem_used_gb"] = used_mb / 1024.0
        out["gpu_mem_total_gb"] = total_mb / 1024.0
        out["gpu_mem_util_pct"] = 100.0 * used_mb / max(total_mb, 1.0)
        return out
    except Exception:
        return out

class ValidationResidualAndLRMetrics(Callback):
    """
    Computes extra validation metrics per epoch and logs them via pl_module.log so they appear in
    trainer.callback_metrics (and thus can be written into metrics.csv and monitored by EarlyStopping).

    NOTE: This does an extra forward pass over (a subset of) val_loader each epoch.
    max_batches controls cost. None = full val set.
    """

    def __init__(self, target: str, val_loader, max_batches: Optional[int] = None):
        self.target = target
        self.val_loader = val_loader
        self.max_batches = max_batches

    @staticmethod
    def _quantiles_and_W(x: torch.Tensor):
        if x.numel() == 0:
            nan = float("nan")
            return nan, nan, nan, nan
        qs = torch.tensor([0.16, 0.50, 0.84], dtype=torch.float32)
        p16, p50, p84 = torch.quantile(x.to(torch.float32), qs)
        W = (p84 - p16) / 2.0
        return float(p16.item()), float(p50.item()), float(p84.item()), float(W.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        # --- LR at end of epoch ---
        lr = float("nan")
        try:
            if trainer.optimizers:
                lr = float(trainer.optimizers[0].param_groups[0].get("lr", float("nan")))
        except Exception:
            pass
        pl_module.log("lr", lr, on_step=False, on_epoch=True, prog_bar=False, logger=False)

        device = pl_module.device

        # ---- preserve training/eval mode ----
        was_training = pl_module.training
        pl_module.eval()

        residual_deg_all = []
        kappa_all = []
        residual_log10_all = []

        with torch.no_grad():
            for ib, batch in enumerate(self.val_loader):
                if self.max_batches is not None and ib >= self.max_batches:
                    break

                batch = move_batch_to_device(batch, device)

                preds_list = pl_module(batch)
                pred0 = preds_list[0].detach().float()

                if self.target in ["zenith", "azimuth"]:
                    pred_angle = pred0[:, 0]
                    pred_kappa = pred0[:, 1]

                    truth = extract_field(batch, self.target).detach().float().view(-1).to(pred_angle.device)

                    if self.target == "azimuth":
                        residual_rad = _circular_signed_diff(pred_angle, truth)
                    else:
                        residual_rad = (pred_angle - truth)

                    residual_deg = residual_rad * (180.0 / math.pi)

                    residual_deg_all.append(residual_deg.detach().cpu())
                    kappa_all.append(pred_kappa.detach().cpu())

                elif self.target == "energy":
                    pred_log10 = pred0.squeeze(-1)

                    true_E = extract_field(batch, "energy").detach().float().view(-1).to(pred_log10.device)
                    true_log10 = torch.log10(torch.clamp(true_E, min=eps_like(true_E)))

                    residual_log10 = (pred_log10 - true_log10)
                    residual_log10_all.append(residual_log10.detach().cpu())

                else:
                    break

        # ---- restore mode ----
        if was_training:
            pl_module.train()

        # --- Log metrics ---
        if self.target in ["zenith", "azimuth"]:
            residual_deg_all = torch.cat(residual_deg_all, dim=0) if residual_deg_all else torch.empty(0)
            kappa_all = torch.cat(kappa_all, dim=0) if kappa_all else torch.empty(0)

            p16, p50, p84, W = self._quantiles_and_W(residual_deg_all)
            pl_module.log("val_residual_p16_deg", p16, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            pl_module.log("val_residual_p50_deg", p50, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            pl_module.log("val_residual_p84_deg", p84, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            pl_module.log("val_W_deg", W, on_step=False, on_epoch=True, prog_bar=False, logger=False)

            kp16, kp50, kp84, kW = self._quantiles_and_W(kappa_all)
            pl_module.log("val_kappa_p16", kp16, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            pl_module.log("val_kappa_p50", kp50, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            pl_module.log("val_kappa_p84", kp84, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            pl_module.log("val_kappa_W", kW, on_step=False, on_epoch=True, prog_bar=False, logger=False)

        elif self.target == "energy":
            residual_log10_all = torch.cat(residual_log10_all, dim=0) if residual_log10_all else torch.empty(0)

            p16, p50, p84, W = self._quantiles_and_W(residual_log10_all)
            pl_module.log("val_residual_log10_p16", p16, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            pl_module.log("val_residual_log10_p50", p50, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            pl_module.log("val_residual_log10_p84", p84, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            pl_module.log("val_W_log10", W, on_step=False, on_epoch=True, prog_bar=False, logger=False)

            if residual_log10_all.numel() > 0:
                mae = float(residual_log10_all.abs().mean().item())
                rmse = float(torch.sqrt((residual_log10_all ** 2).mean()).item())
            else:
                mae = float("nan")
                rmse = float("nan")
            if residual_log10_all.numel() > 0:
                bias = float(residual_log10_all.mean().item())
            else:
                bias = float("nan")

            pl_module.log("val_bias_log10", bias, on_step=False, on_epoch=True, prog_bar=False, logger=False)


            pl_module.log("val_mae_log10", mae, on_step=False, on_epoch=True, prog_bar=False, logger=False)
            pl_module.log("val_rmse_log10", rmse, on_step=False, on_epoch=True, prog_bar=False, logger=False)


class EpochTimeLogger(Callback):
    """
    Logs per-epoch timing + resource snapshots to a single CSV.

    Output CSV columns:
      epoch, elapsed_min, epoch_duration_min, rss_gb, sys_mem_used_gb, cpu_load_pct,
      gpu_util_pct, gpu_mem_used_gb, gpu_mem_total_gb, gpu_mem_util_pct

    Behavior:
      - At fit start: writes header only.
      - At each validation epoch end: appends one row for the completed epoch.
        (So 'epoch' matches metrics.csv written at on_validation_epoch_end.)
      - At fit end: prints a one-line summary with file path and total time.
    """

    def __init__(self, out_dir, filename: str = "epoch_time.csv"):
        self.out_dir = Path(out_dir)
        self.file = self.out_dir / filename
        self.t0 = None
        self.prev_elapsed_min = 0.0 
        self.fieldnames = [
            "epoch",
            "elapsed_min",
            "epoch_duration_min",
            "rss_gb",
            "sys_mem_used_gb",
            "cpu_load_pct",
            "gpu_util_pct",
            "gpu_mem_used_gb",
            "gpu_mem_total_gb",
            "gpu_mem_util_pct",
        ]

    def _snapshot_row(self, epoch: int, elapsed_min: float, epoch_dur_min: float) -> Dict[str, object]:
        gpu = _gpu_snapshot()
        return {
            "epoch": epoch,
            "elapsed_min": f"{elapsed_min:.3f}",
            "epoch_duration_min": f"{epoch_dur_min:.3f}",
            "rss_gb": f"{_read_rss_gb():.3f}",
            "sys_mem_used_gb": f"{_read_system_mem_used_gb():.3f}",
            "cpu_load_pct": f"{_cpu_load_pct():.2f}",
            "gpu_util_pct": f"{gpu['gpu_util_pct']:.1f}",
            "gpu_mem_used_gb": f"{gpu['gpu_mem_used_gb']:.3f}",
            "gpu_mem_total_gb": f"{gpu['gpu_mem_total_gb']:.3f}",
            "gpu_mem_util_pct": f"{gpu['gpu_mem_util_pct']:.1f}",
        }

    
    def on_fit_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        self.t0 = time.time()
        self.prev_elapsed_min = 0.0
        self.last_written_epoch = None

        self.out_dir.mkdir(parents=True, exist_ok=True)
        with self.file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.t0 is None:
            return

        ep = trainer.current_epoch
        elapsed_min = (time.time() - self.t0) / 60.0
        epoch_dur_min = elapsed_min - self.prev_elapsed_min

        with self.file.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(self._snapshot_row(ep, elapsed_min, epoch_dur_min))

        self.prev_elapsed_min = elapsed_min
        self.last_written_epoch = ep
    
    def on_fit_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.t0 is None:
            return
    
        total_min = (time.time() - self.t0) / 60.0
        print(f"[Resources] Wrote {self.file} | last_epoch={self.last_written_epoch} | total={total_min:.2f} min")

class EpochCSVLogger(Callback):
    def __init__(self, out_dir, filename: str = "metrics.csv"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.file = self.out_dir / filename

        # --- Track best_model.pth updates per epoch (robust signature) ---
        self.best_model_path = self.out_dir / "best_model.pth"
        self._best_sig_prev: Optional[Tuple[int, int]] = None  # (mtime_ns, size)
        self._last_epoch_written: Optional[int] = None

        self.extra_keys_in_order = [
            "val_residual_p16_deg",
            "val_residual_p50_deg",
            "val_residual_p84_deg",
            "val_W_deg",
            "val_kappa_p16",
            "val_kappa_p50",
            "val_kappa_p84",
            "val_kappa_W",
            "val_residual_log10_p16",
            "val_residual_log10_p50",
            "val_residual_log10_p84",
            "val_W_log10",
            "val_mae_log10",
            "val_rmse_log10",
        ]

    def _best_sig(self) -> Optional[Tuple[int, int]]:
        try:
            st = self.best_model_path.stat()
            mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
            return (int(mtime_ns), int(st.st_size))
        except FileNotFoundError:
            return None

    def on_fit_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._best_sig_prev = self._best_sig()
        self._last_epoch_written = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics

        cur_sig = self._best_sig()
        updated_now = (cur_sig is not None) and (
            self._best_sig_prev is None or cur_sig != self._best_sig_prev
        )

        row: Dict[str, object] = {
            "epoch": trainer.current_epoch,
            "train_loss": metrics.get("train_loss_epoch", metrics.get("train_loss")),
            "val_loss": metrics.get("val_loss"),
            "lr": metrics.get("lr", float("nan")),
            "best_model_is_updated": bool(updated_now),
        }

        for k in self.extra_keys_in_order:
            row[k] = metrics.get(k, float("nan"))

        for k, v in list(row.items()):
            if torch.is_tensor(v):
                row[k] = v.item()

        write_header = not self.file.exists()
        with self.file.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self._last_epoch_written = trainer.current_epoch

        # Update baseline *after* we wrote the row (so next epoch compares correctly)
        if cur_sig is not None:
            self._best_sig_prev = cur_sig

    def _patch_last_row_best_flag(self, epoch: int) -> None:
        """Patch the last row in metrics.csv for the given epoch -> set best_model_is_updated=True."""
        if not self.file.exists():
            return

        with self.file.open("r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        if not rows or "best_model_is_updated" not in fieldnames:
            return

        for r in reversed(rows):
            if str(r.get("epoch", "")) == str(epoch):
                r["best_model_is_updated"] = "True"
                break

        tmp = self.file.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        tmp.replace(self.file)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        # Catch updates that happen after on_validation_epoch_end
        cur_sig = self._best_sig()
        updated_late = (cur_sig is not None) and (
            self._best_sig_prev is None or cur_sig != self._best_sig_prev
        )

        if updated_late and self._last_epoch_written == trainer.current_epoch:
            self._patch_last_row_best_flag(trainer.current_epoch)

        if cur_sig is not None:
            self._best_sig_prev = cur_sig


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


def _wrap_to_pi(d: torch.Tensor) -> torch.Tensor:
    """Wrap angle differences to [-pi, +pi]."""
    period = 2 * math.pi
    return torch.remainder(d + period / 2, period) - period / 2

def _circular_signed_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Signed circular difference a-b in [-pi, pi]."""
    return _wrap_to_pi(a - b)


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
    save_dir: str = "/project/6061446/kbas/Graphnet-Applications/Training/EventPulseSeries_nonoise/initiative2"
    metrics_name: str = "metrics.csv"
    test_csv_name: str = "test_predictions.csv"
    resources_and_time_csv_name: str = "resources_and_time.csv"   

    # Extra metric columns to write into metrics.csv (chosen by target)
    metrics_extra_keys_energy: Tuple[str, ...] = (
        "val_residual_log10_p16",
        "val_residual_log10_p50",
        "val_residual_log10_p84",
        "val_W_log10",
        "val_bias_log10",
        "val_mae_log10",
        "val_rmse_log10",
    )
    metrics_extra_keys_zenith: Tuple[str, ...] = (
        "val_residual_p16_deg",
        "val_residual_p50_deg",
        "val_residual_p84_deg",
        "val_W_deg",
        "val_kappa_p16",
        "val_kappa_p50",
        "val_kappa_p84",
        "val_kappa_W",
    )
    metrics_extra_keys_azimuth: Tuple[str, ...] = (
        "val_residual_p16_deg",
        "val_residual_p50_deg",
        "val_residual_p84_deg",
        "val_W_deg",
        "val_kappa_p16",
        "val_kappa_p50",
        "val_kappa_p84",
        "val_kappa_W",
    )

    # Early stopping monitors (separate)
    early_stopping_monitor_energy: str = "val_loss"
    early_stopping_monitor_angle: str = "val_loss"
    early_stopping_mode: str = "min"

    # Optional: speed control for extra val-metrics pass
    val_metrics_max_batches: Optional[int] = None  # e.g. 50 to speed up; None = use all




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

            def _eid(i):
                if event_id is not None and i < len(event_id):
                    return int(event_id[i].item())
                return None

            if target in ["zenith", "azimuth"]:
                if pred0.ndim != 2 or pred0.shape[1] != 2:
                    raise RuntimeError(f"[Test={target}] Expected pred [N,2], got {tuple(pred0.shape)}")

                pred_angle = pred0[:, 0].cpu()
                kappa = pred0[:, 1].cpu()
                truth = extract_field(batch, target).detach().float().view(-1).cpu()

                if target == "azimuth":
                    residual_rad = _circular_signed_diff(pred_angle, truth)
                else:
                    residual_rad = (pred_angle - truth)

                residual_deg = residual_rad * (180.0 / math.pi)

                true_deg = truth * (180.0 / math.pi)
                pred_deg = pred_angle * (180.0 / math.pi)

                if target == "azimuth":
                    true_signed_rad = _wrap_to_pi(truth)
                    pred_signed_rad = _wrap_to_pi(pred_angle)
                    true_signed_deg = true_signed_rad * (180.0 / math.pi)
                    pred_signed_deg = pred_signed_rad * (180.0 / math.pi)
                    pred_adj_deg = true_deg + residual_deg

                for i in range(len(truth)):
                    if target == "zenith":
                        row = {
                            "true_zenith_radian": float(truth[i].item()),
                            "pred_zenith_radian": float(pred_angle[i].item()),
                            "true_zenith_degree": float(true_deg[i].item()),
                            "pred_zenith_degree": float(pred_deg[i].item()),
                            "kappa": float(kappa[i].item()),
                            "event_id": _eid(i),
                            "residual_zenith_radian": float(residual_rad[i].item()),
                            "residual_zenith_degree": float(residual_deg[i].item()),
                        }
                    else:
                        row = {
                            "true_azimuth_radian": float(truth[i].item()),
                            "pred_azimuth_radian": float(pred_angle[i].item()),
                            "true_azimuth_degree": float(true_deg[i].item()),
                            "pred_azimuth_degree": float(pred_deg[i].item()),
                            "true_azimuth_degree_signed": float(true_signed_deg[i].item()),
                            "pred_azimuth_degree_signed": float(pred_signed_deg[i].item()),
                            "pred_azimuth_degree_adj": float(pred_adj_deg[i].item()),
                            "kappa": float(kappa[i].item()),
                            "event_id": _eid(i),
                            "residual_azimuth_radian": float(residual_rad[i].item()),
                            "residual_azimuth_degree": float(residual_deg[i].item()),
                        }
                    rows.append(row)

            else:
                pred_log10 = pred0.squeeze(-1).cpu()
                true_E = extract_field(batch, "energy").detach().float().view(-1).cpu()

                true_log10 = logarithm(true_E)
                pred_E = exponential(pred_log10)

                residual_log10 = (pred_log10 - true_log10)
                residual = (pred_E - true_E)

                for i in range(len(pred_log10)):
                    row = {
                        "true_energy": float(true_E[i].item()),
                        "pred_energy": float(pred_E[i].item()),
                        "pred_log10_energy": float(pred_log10[i].item()),
                        "true_log10_energy": float(true_log10[i].item()),
                        "residual_log10": float(residual_log10[i].item()),
                        "residual": float(residual[i].item()),
                        "event_id": _eid(i),
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)

    if target == "energy":
        preferred = [
            "true_energy",
            "pred_energy",
            "pred_log10_energy",
            "true_log10_energy",
            "residual_log10",
            "residual",
            "event_id",
        ]
        cols = preferred + [c for c in df.columns if c not in preferred]
        df = df[cols]
    elif target == "zenith":
        preferred = [
            "true_zenith_radian",
            "pred_zenith_radian",
            "true_zenith_degree",
            "pred_zenith_degree",
            "kappa",
            "event_id",
            "residual_zenith_radian",
            "residual_zenith_degree",
        ]
        cols = preferred + [c for c in df.columns if c not in preferred]
        df = df[cols]
    elif target == "azimuth":
        preferred = [
            "true_azimuth_radian",
            "pred_azimuth_radian",
            "true_azimuth_degree",
            "pred_azimuth_degree",
            "true_azimuth_degree_signed",
            "pred_azimuth_degree_signed",
            "pred_azimuth_degree_adj",
            "kappa",
            "event_id",
            "residual_azimuth_radian",
            "residual_azimuth_degree",
        ]
        cols = preferred + [c for c in df.columns if c not in preferred]
        df = df[cols]

    df.to_csv(out_csv, index=False)
    print(f"[Test={target}] Wrote: {out_csv} | rows={len(df)}")

    # Quick summary: ONLY signed residual stats (no abs error)
    if target in ["zenith", "azimuth"]:
        if target == "zenith":
            rdeg = torch.tensor(df["residual_zenith_degree"].to_numpy(), dtype=torch.float32)
        else:
            rdeg = torch.tensor(df["residual_azimuth_degree"].to_numpy(), dtype=torch.float32)

        rp16 = torch.quantile(rdeg, 0.16).item()
        rp50 = torch.quantile(rdeg, 0.50).item()
        rp84 = torch.quantile(rdeg, 0.84).item()
        Wdeg = (rp84 - rp16) / 2.0
        kmean = float(df["kappa"].mean())
        print(f"[Test={target}] residual_deg (signed): p16={rp16:.3f}, p50={rp50:.3f}, p84={rp84:.3f} | W_deg={Wdeg:.3f} | kappa_mean={kmean:.3f}")
    else:
        r = torch.tensor(df["residual_log10"].to_numpy(), dtype=torch.float32)
        p16 = torch.quantile(r, 0.16).item()
        p50 = torch.quantile(r, 0.50).item()
        p84 = torch.quantile(r, 0.84).item()
        W = (p84 - p16) / 2.0
        mae = r.abs().mean().item()
        rmse = torch.sqrt((r ** 2).mean()).item()
        print(f"[Test=energy] residual_log10: p16={p16:.4f}, p50={p50:.4f}, p84={p84:.4f}, W={W:.4f} | mae={mae:.4f} rmse={rmse:.4f}")


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


    monitor = cfg.early_stopping_monitor_energy if target == "energy" else cfg.early_stopping_monitor_angle

    early_stop = GraphnetEarlyStopping(
        save_dir=out_dir,
        monitor=monitor,
        mode=cfg.early_stopping_mode,
        patience=cfg.early_stopping_patience,
        check_on_train_epoch_end=False,
        verbose=True,
    )



    metrics_cb = EpochCSVLogger(out_dir, filename=cfg.metrics_name)
    # Select per-target metric columns (avoid NaNs in metrics.csv)
    if target == "energy":
        metrics_cb.extra_keys_in_order = list(cfg.metrics_extra_keys_energy)
    elif target == "zenith":
        metrics_cb.extra_keys_in_order = list(cfg.metrics_extra_keys_zenith)
    elif target == "azimuth":
        metrics_cb.extra_keys_in_order = list(cfg.metrics_extra_keys_azimuth)

    epoch_ctx_cb = _EpochContextCallback()
    time_cb = EpochTimeLogger(out_dir, filename=cfg.resources_and_time_csv_name)

    val_metrics_cb = ValidationResidualAndLRMetrics(
        target=target,
        val_loader=val_loader,
        max_batches=cfg.val_metrics_max_batches,
        )


    # --- GPU sanity print
    print(f"\n[Run={target}] torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Run={target}] GPU = {torch.cuda.get_device_name(0)}")


    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[epoch_ctx_cb, val_metrics_cb, early_stop, metrics_cb, time_cb],
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

    for target in ["energy",  "azimuth" , "zenith"]:
        run_one(cfg, target, data_representation, train_loader, val_loader, test_loader)
