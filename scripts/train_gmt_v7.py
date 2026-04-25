#!/usr/bin/env python3
"""Training entrypoint for the base GMT v7 model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

from gmt import GMTTrainingConfig, GMTV7, GMTV7Config, Trainer, build_dataloaders


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {path}")
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gmt_v7_base.yaml"),
        help="YAML configuration file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing train.bin and val.bin; overrides the config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints and progress logs; overrides the config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-step dataloader batch size.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", dest="compile_model", action="store_true")
    parser.add_argument("--no-compile", dest="compile_model", action="store_false")
    parser.set_defaults(compile_model=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run validation and exit. Use with --resume to evaluate a checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("gmt.train")

    args = parse_args()
    cfg = load_yaml(args.config)

    model_cfg = GMTV7Config(**cfg.get("model", {}))
    training_cfg = GMTTrainingConfig(**cfg.get("training", {}))

    if args.data_dir is not None:
        training_cfg.data_dir = args.data_dir
    if args.output_dir is not None:
        training_cfg.output_dir = args.output_dir
    if args.epochs is not None:
        training_cfg.epochs = args.epochs
    if args.batch_size is not None:
        training_cfg.batch_size = args.batch_size
    if args.no_amp:
        training_cfg.use_amp = False
    if args.compile_model is not None:
        training_cfg.use_compile = args.compile_model
    if args.eval_only and not args.resume:
        raise SystemExit("--eval-only requires --resume so a checkpoint is loaded.")

    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)
    logger.info(
        "Architecture: %s x (CausalSelfAttention + GraphMemoryCell)",
        model_cfg.n_layers,
    )
    logger.info(
        "hidden_dim=%s n_heads=%s memory_slots=%s nav_dim=%s",
        model_cfg.hidden_dim,
        model_cfg.n_heads,
        model_cfg.memory_slots,
        model_cfg.nav_dim,
    )

    train_dl, val_dl = build_dataloaders(
        training_cfg.data_dir,
        seq_len=model_cfg.seq_len,
        batch_size=training_cfg.batch_size,
        train_workers=training_cfg.train_workers,
        val_workers=training_cfg.val_workers,
        pin_memory=training_cfg.pin_memory,
    )
    logger.info("Train sequences: %s", len(train_dl.dataset))
    logger.info("Validation sequences: %s", len(val_dl.dataset))

    model = GMTV7(model_cfg)
    logger.info("Parameters: %.2fM", model.num_parameters() / 1e6)
    diag = model.diagnostics()
    model.log_diagnostics(
        diag,
        track=0.0,
        ortho=0.0,
        cluster=0.0,
        edge_ent=0.0,
        contrast=0.0,
        temp=GMTV7.routing_temperature(0, 1, model_cfg.temp_max, model_cfg.temp_min),
    )

    if training_cfg.use_compile:
        if hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile().")
            model = torch.compile(model, mode="default")
        else:
            logger.warning("torch.compile is unavailable; continuing without it.")

    trainer = Trainer(model, train_dl, val_dl, training_cfg, device)
    checkpoint_loaded = False
    if args.resume:
        checkpoint_loaded = trainer.load_checkpoint()
    if args.eval_only:
        if not checkpoint_loaded:
            raise FileNotFoundError(
                f"No checkpoint found in {training_cfg.output_dir} for evaluation."
            )
        val_loss, ppl = trainer.validate(save_best=False)
        logger.info("Validation loss: %.5f | Perplexity: %.2f", val_loss, ppl)
        return
    trainer.fit()


if __name__ == "__main__":
    main()
