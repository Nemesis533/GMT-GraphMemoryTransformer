"""Token-stream datasets for GMT training."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import SEQ_LEN


class TokenStreamDataset(Dataset):
    """Dataset backed by a contiguous uint16 token stream."""

    def __init__(self, bin_path: Union[str, Path], seq_len: int = SEQ_LEN):
        self.path = Path(bin_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Token stream not found: {self.path}")

        self.seq_len = seq_len
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.n_seq = (len(self.data) - 1) // seq_len
        if self.n_seq <= 0:
            raise ValueError(
                f"Token stream {self.path} is too short for seq_len={seq_len}"
            )

    def __len__(self) -> int:
        return self.n_seq

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a DataLoader without enabling persistent workers when unused."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
