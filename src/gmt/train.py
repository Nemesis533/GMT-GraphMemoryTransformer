"""Training utilities for the base GMT v7 model."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

try:
    from torch.amp import GradScaler
except ImportError:  # pragma: no cover - compatibility with older torch builds
    from torch.cuda.amp import GradScaler

from .data import TokenStreamDataset, make_dataloader
from .model import (
    LAMBDA_CLUSTER,
    LAMBDA_CONTRAST,
    LAMBDA_EDGE,
    LAMBDA_TRACK,
    ORTHO_BETA,
    SEQ_LEN,
    GMTV7,
    GMTV7Config,
)

logger = logging.getLogger(__name__)


@dataclass
class GMTTrainingConfig:
    """Training configuration for the base v7 GMT run."""

    data_dir: Union[str, Path] = Path("data/prepared_owt")
    output_dir: Union[str, Path] = Path("runs/gmt_v7_base")
    epochs: int = 2
    batch_size: int = 8
    grad_accum: int = 33
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 2000
    max_val_batches: int = 512
    train_workers: int = 4
    val_workers: int = 2
    pin_memory: bool = True
    use_amp: bool = True
    use_compile: bool = True
    lambda_track: float = LAMBDA_TRACK
    ortho_beta: float = ORTHO_BETA
    lambda_cluster: float = LAMBDA_CLUSTER
    lambda_edge: float = LAMBDA_EDGE
    lambda_contrast: float = LAMBDA_CONTRAST
    merge_every: int = 110
    save_every: int = 200

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _make_grad_scaler(device_type: str, enabled: bool):
    try:
        return GradScaler(device_type, enabled=enabled)
    except TypeError:
        return GradScaler(enabled=enabled)


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def build_dataloaders(
    data_dir: Union[str, Path],
    seq_len: int = SEQ_LEN,
    batch_size: int = 8,
    train_workers: int = 4,
    val_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir)
    expected = (data_dir / "train.bin", data_dir / "val.bin")
    missing = [path for path in expected if not path.exists()]
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"GMT training data is incomplete in {data_dir}. "
            f"Expected train.bin and val.bin; missing: {missing_names}"
        )

    train_ds = TokenStreamDataset(data_dir / "train.bin", seq_len=seq_len)
    val_ds = TokenStreamDataset(data_dir / "val.bin", seq_len=seq_len)
    train_dl = make_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=pin_memory,
    )
    val_dl = make_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=pin_memory,
    )
    return train_dl, val_dl


class Trainer:
    """Trainer for next-token language modeling with GMT auxiliary losses."""

    def __init__(
        self,
        model: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        config: GMTTrainingConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.config = config
        self.device = device
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.grad_accum = config.grad_accum
        self.use_amp = config.use_amp and device.type == "cuda"
        self.scaler = _make_grad_scaler(device.type, enabled=self.use_amp)
        self.loss_fn = nn.CrossEntropyLoss()

        self.opt = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
        total_steps = max((len(train_dl) // config.grad_accum) * config.epochs, 1)
        self.total_steps = total_steps
        self.sched = get_cosine_schedule_with_warmup(
            self.opt,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
        )

        self.best_loss = float("inf")
        self.global_step = 0
        self._epoch = 0
        self._batch_idx = 0
        self._resume_epoch = 0
        self._resume_batch = 0
        self._reset_loss_accumulators()
        self._init_progress_csv()

    def _reset_loss_accumulators(self) -> None:
        self._track_acc = 0.0
        self._ortho_acc = 0.0
        self._cluster_acc = 0.0
        self._edge_acc = 0.0
        self._contrast_acc = 0.0
        self._temp_acc = 0.0
        self._accum_n = 0

    def _init_progress_csv(self) -> None:
        self.csv_path = self.output_dir / "progress.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "step",
                    "total_steps",
                    "epoch",
                    "pct",
                    "val_loss",
                    "ppl",
                    "N_eff_mean",
                    "N_eff_min",
                    "dead_total",
                    "cos_sim_mean",
                    "coverage_mean",
                    "gate_mean",
                    "gate_min",
                    "gate_max",
                    "momentum_mean",
                    "momentum_min",
                    "momentum_max",
                    "edge_ent_mean",
                    "edge_max_mean",
                    "e_row_sim_mean",
                    "routing_temp",
                    "track_loss",
                    "ortho_loss",
                    "cluster_penalty",
                    "edge_ent_loss",
                    "contrast_loss",
                ]
            )

    def load_checkpoint(self) -> None:
        latest = self.output_dir / "latest.pt"
        best = self.output_dir / "best.pt"
        path = latest if latest.exists() else (best if best.exists() else None)
        if path is None:
            logger.info("No checkpoint found; starting from scratch.")
            return

        logger.info("Loading checkpoint: %s", path)
        ckpt = torch.load(path, map_location=self.device)
        raw = unwrap_model(self.model)
        raw.load_state_dict(ckpt["model"])
        self.global_step = ckpt["step"]
        self.best_loss = ckpt.get("loss", float("inf"))
        self._resume_epoch = ckpt.get("epoch", 0)
        self._resume_batch = ckpt.get("batch_idx", 0)
        if "opt" in ckpt:
            self.opt.load_state_dict(ckpt["opt"])
        if "sched" in ckpt:
            self.sched.load_state_dict(ckpt["sched"])
        else:
            for _ in range(self.global_step):
                self.sched.step()
        logger.info("Resumed at step=%s loss=%.4f", self.global_step, self.best_loss)

    def _save_checkpoint(self, name: str, loss: Optional[float] = None) -> None:
        raw = unwrap_model(self.model)
        torch.save(
            {
                "model": raw.state_dict(),
                "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(),
                "step": self.global_step,
                "loss": self.best_loss if loss is None else loss,
                "epoch": self._epoch,
                "batch_idx": self._batch_idx,
            },
            self.output_dir / name,
        )

    def _write_progress_row(
        self,
        pct: int,
        val_loss: float,
        ppl: float,
        diag,
        track: float,
        ortho: float,
        cluster: float,
        edge: float,
        contrast: float,
        temp: float,
    ) -> None:
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    self.global_step,
                    self.total_steps,
                    self._epoch,
                    pct,
                    f"{val_loss:.5f}",
                    f"{ppl:.3f}",
                    f"{diag['N_eff_mean']:.1f}",
                    f"{diag['N_eff_min']:.1f}",
                    diag["dead_total"],
                    f"{diag['cos_sim_mean']:.4f}",
                    f"{diag['coverage_mean']:.4f}",
                    f"{diag['gate_mean']:.4f}",
                    f"{diag['gate_min']:.4f}",
                    f"{diag['gate_max']:.4f}",
                    f"{diag['momentum_mean']:.4f}",
                    f"{diag['momentum_min']:.4f}",
                    f"{diag['momentum_max']:.4f}",
                    f"{diag['edge_ent_mean']:.4f}",
                    f"{diag['edge_max_mean']:.4f}",
                    f"{diag['e_row_sim_mean']:.4f}",
                    f"{temp:.4f}",
                    f"{track:.5f}",
                    f"{ortho:.5f}",
                    f"{cluster:.5f}",
                    f"{edge:.5f}",
                    f"{contrast:.5f}",
                ]
            )

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        self.opt.zero_grad()
        self._reset_loss_accumulators()

        skip = self._resume_batch if self._epoch == self._resume_epoch else 0
        if skip:
            logger.info("Skipping %s resumed batches", skip)

        report_interval = max(len(self.train_dl) // 20, 1)
        bar = tqdm(
            self.train_dl,
            desc="Train",
            dynamic_ncols=True,
            total=len(self.train_dl),
            initial=skip,
            postfix={"step": f"{self.global_step}/{self.total_steps}"},
        )

        for i, (x, y) in enumerate(bar):
            if i < skip:
                continue
            self._batch_idx = i
            x = x.to(self.device)
            y = y.to(self.device)

            raw_model = unwrap_model(self.model)
            temp = raw_model.routing_temperature(
                self.global_step,
                self.total_steps,
                raw_model.config.temp_max,
                raw_model.config.temp_min,
            )
            raw_model.temp.fill_(temp)

            with autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.use_amp,
            ):
                logits, L_track, L_ortho, L_cluster, L_edge, L_contrast, h = (
                    self.model(x, return_mem_loss=True)
                )
                task_loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                mem_loss = (
                    self.config.lambda_track * L_track
                    + self.config.ortho_beta * L_ortho
                    + self.config.lambda_cluster * L_cluster
                    + self.config.lambda_edge * L_edge
                    + self.config.lambda_contrast * L_contrast
                )
                loss = (task_loss + mem_loss) / self.grad_accum

            self.scaler.scale(loss).backward()
            self._track_acc += L_track.item()
            self._ortho_acc += L_ortho.item()
            self._cluster_acc += L_cluster.item()
            self._edge_acc += L_edge.item()
            self._contrast_acc += L_contrast.item()
            self._temp_acc += temp
            self._accum_n += 1

            if (i + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.sched.step()
                self.opt.zero_grad()
                self.global_step += 1

                raw = unwrap_model(self.model)
                if self.global_step % self.config.merge_every == 0:
                    total_dead = 0
                    total_merged = 0
                    for block in raw.blocks:
                        total_dead += block.cell.reset_dead_centroids(h)
                        total_merged += block.cell.merge_similar_centroids(h)
                    if total_dead > 0 or total_merged > 0:
                        logger.info(
                            "Maintenance at step %s: reset=%s merged=%s",
                            self.global_step,
                            total_dead,
                            total_merged,
                        )

                if self.global_step % self.config.save_every == 0:
                    self._save_checkpoint("latest.pt")

            task_l = task_loss.item()
            total_loss += task_l
            bar.set_postfix(
                loss=f"{task_l:.4f}",
                ctr=f"{L_contrast.item():.3f}",
                mom=(
                    f"{torch.sigmoid(raw_model.blocks[0].cell.write_momentum).item():.3f}"
                ),
                temp=f"{temp:.3f}",
                step=f"{self.global_step}/{self.total_steps}",
            )

            if (i + 1) % report_interval == 0:
                raw = unwrap_model(self.model)
                diag = raw.diagnostics()
                val_loss, ppl = self.validate()
                pct = int((i + 1) / len(self.train_dl) * 100)
                n = max(self._accum_n, 1)
                track = self._track_acc / n
                ortho = self._ortho_acc / n
                cluster = self._cluster_acc / n
                edge = self._edge_acc / n
                contrast = self._contrast_acc / n
                temp_avg = self._temp_acc / n

                logger.info(
                    "[%s%%] Val: %.4f | PPL: %.2f | Step: %s",
                    pct,
                    val_loss,
                    ppl,
                    self.global_step,
                )
                raw.log_diagnostics(
                    diag, track, ortho, cluster, edge, contrast, temp_avg
                )
                self._write_progress_row(
                    pct,
                    val_loss,
                    ppl,
                    diag,
                    track,
                    ortho,
                    cluster,
                    edge,
                    contrast,
                    temp_avg,
                )
                self._reset_loss_accumulators()
                self.model.train()

        return total_loss / len(self.train_dl)

    def validate(self, max_batches: Optional[int] = None) -> Tuple[float, float]:
        self.model.eval()
        total = 0.0
        n = 0
        limit = self.config.max_val_batches if max_batches is None else max_batches
        with torch.no_grad():
            for x, y in tqdm(
                self.val_dl,
                desc="Val",
                dynamic_ncols=True,
                total=min(limit, len(self.val_dl)),
            ):
                x = x.to(self.device)
                y = y.to(self.device)
                with autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.use_amp,
                ):
                    logits, _ = self.model(x, return_mem_loss=False)
                    total += self.loss_fn(
                        logits.view(-1, logits.size(-1)), y.view(-1)
                    ).item()
                n += 1
                if n >= limit:
                    break

        if n == 0:
            raise RuntimeError("Validation dataloader produced no batches")

        loss = total / n
        ppl = float(np.exp(min(loss, 20)))
        if loss < self.best_loss:
            self.best_loss = loss
            self._save_checkpoint("best.pt", loss=loss)
            logger.info("New best validation loss: %.5f", loss)
        return loss, ppl

    def fit(self) -> None:
        for ep in range(self.config.epochs):
            self._epoch = ep + 1
            logger.info("Epoch %s/%s", ep + 1, self.config.epochs)
            train_loss = self.train_epoch()
            val_loss, ppl = self.validate()
            logger.info(
                "Train: %.4f | Val: %.4f | PPL: %.2f",
                train_loss,
                val_loss,
                ppl,
            )


def build_model(config: Optional[GMTV7Config] = None) -> GMTV7:
    return GMTV7(config or GMTV7Config())
