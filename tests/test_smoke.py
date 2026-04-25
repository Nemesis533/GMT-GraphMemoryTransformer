"""Portable smoke tests for the GMT package."""

import numpy as np
import torch

from gmt import GMTV7, GMTV7Config, GraphMemoryCell, TokenStreamDataset


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
