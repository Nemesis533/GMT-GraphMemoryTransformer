"""Portable smoke tests for the GMT package."""

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import torch

from gmt import (
    GMTTrainingConfig,
    GMTV7,
    GMTV7Config,
    GraphMemoryCell,
    TokenStreamDataset,
    Trainer,
    build_dataloaders,
    build_model,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def tiny_config() -> GMTV7Config:
    return GMTV7Config(
        vocab_size=64,
        seq_len=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=2,
        memory_slots=8,
        nav_dim=4,
        attention_dropout=0.0,
        embedding_dropout=0.0,
    )


def load_training_script():
    script_path = REPO_ROOT / "scripts" / "train_gmt_v7.py"
    spec = importlib.util.spec_from_file_location("train_gmt_v7", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_imports_public_api() -> None:
    package = importlib.import_module("gmt")

    assert package.GMTV7 is GMTV7
    assert package.GMTV7Config is GMTV7Config
    assert package.GMTTrainingConfig is GMTTrainingConfig


def test_default_yaml_config_loads() -> None:
    train_script = load_training_script()
    cfg = train_script.load_yaml(REPO_ROOT / "configs" / "gmt_v7_base.yaml")

    model_cfg = GMTV7Config(**cfg["model"])
    training_cfg = GMTTrainingConfig(**cfg["training"])

    assert model_cfg.memory_slots == 128
    assert model_cfg.n_layers == 16
    assert str(training_cfg.data_dir) == "data/prepared_owt"
    assert str(training_cfg.output_dir) == "runs/gmt_v7_base"


def test_model_constructs_from_package_helper() -> None:
    model = build_model(tiny_config())

    assert isinstance(model, GMTV7)
    assert len(model.blocks) == 2


def test_model_forward_shapes_and_losses() -> None:
    torch.manual_seed(0)
    model = GMTV7(tiny_config())
    x = torch.randint(0, model.config.vocab_size, (2, model.config.seq_len))

    logits, track, ortho, cluster, edge, contrast, h = model(
        x, return_mem_loss=True
    )

    assert logits.shape == (2, model.config.seq_len, model.config.vocab_size)
    assert h.shape == (2, model.config.seq_len, model.config.hidden_dim)
    for loss in (track, ortho, cluster, edge, contrast):
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_weight_tying_and_temperature_schedule() -> None:
    model = GMTV7(tiny_config())

    assert model.lm_head.weight is model.embed.weight
    assert GMTV7.routing_temperature(0, 100) == 1.0
    assert abs(GMTV7.routing_temperature(100, 100) - 0.1) < 1e-7


def test_graph_memory_cell_hard_write_back_updates_centroids() -> None:
    torch.manual_seed(0)
    cell = GraphMemoryCell(hidden_dim=4, n_slots=3, nav_dim=2, block_idx=0)
    before = cell.C.detach().clone()
    h = torch.randn(2, 2, 4)
    w_src = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )

    cell.write_back(h, w_src)

    assert not torch.allclose(before, cell.C)
    assert torch.allclose(cell.C.norm(dim=-1), torch.ones(3), atol=1e-6)
    assert torch.equal(cell.centroid_age, torch.ones(3))


def test_token_stream_dataset(tmp_path) -> None:
    path = tmp_path / "train.bin"
    np.asarray([1, 2, 3, 4, 5], dtype=np.uint16).tofile(path)

    dataset = TokenStreamDataset(path, seq_len=2)

    assert len(dataset) == 2
    x, y = dataset[1]
    assert x.tolist() == [3, 4]
    assert y.tolist() == [4, 5]


def test_training_config_uses_portable_defaults() -> None:
    config = GMTTrainingConfig()

    assert str(config.data_dir) == "data/prepared_owt"
    assert str(config.output_dir) == "runs/gmt_v7_base"
    assert config.batch_size == 8
    assert config.grad_accum == 33


def test_build_dataloaders_reports_missing_streams(tmp_path) -> None:
    try:
        build_dataloaders(
            tmp_path,
            seq_len=2,
            batch_size=1,
            train_workers=0,
            val_workers=0,
        )
    except FileNotFoundError as exc:
        message = str(exc)
    else:  # pragma: no cover - failure branch for readability
        raise AssertionError(
            "build_dataloaders should fail when token streams are absent"
        )

    assert "Expected train.bin and val.bin" in message
    assert "train.bin" in message
    assert "val.bin" in message


def test_tiny_synthetic_training_step(tmp_path) -> None:
    torch.manual_seed(0)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.arange(40, dtype=np.uint16).tofile(data_dir / "train.bin")
    np.arange(20, dtype=np.uint16).tofile(data_dir / "val.bin")

    model_cfg = tiny_config()
    train_dl, val_dl = build_dataloaders(
        data_dir,
        seq_len=model_cfg.seq_len,
        batch_size=2,
        train_workers=0,
        val_workers=0,
        pin_memory=False,
    )
    training_cfg = GMTTrainingConfig(
        data_dir=data_dir,
        output_dir=tmp_path / "runs",
        epochs=1,
        batch_size=2,
        grad_accum=1,
        learning_rate=1e-3,
        warmup_steps=0,
        max_val_batches=1,
        train_workers=0,
        val_workers=0,
        pin_memory=False,
        use_amp=False,
        use_compile=False,
        merge_every=1000,
        save_every=1000,
    )
    trainer = Trainer(
        GMTV7(model_cfg),
        train_dl,
        val_dl,
        training_cfg,
        torch.device("cpu"),
    )

    train_loss = trainer.train_epoch()

    assert np.isfinite(train_loss)
    assert trainer.global_step > 0
    assert (training_cfg.output_dir / "progress.csv").exists()
