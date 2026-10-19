"""Packet storage.

A packet is a tensor [P, S, d]. It depends only on the context, so it is computed once,
stored, and reused for every question about that context. Packets are keyed by a
hash of the context text; identical contexts share one stored packet.

A packet directory holds

    meta.json            packet shape and storage type
    index-rank*.json     context key -> [shard file, row]
    packets-rank*-*.npy  shards of stored packets

Packets are stored as 32-bit floats or as bfloat16. NumPy has no bfloat16 type, so a
bfloat16 shard keeps the 16-bit patterns in an int16 array. The readers always receive
32-bit floats.
"""
from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from latcomm.data import Example

SHARD_SIZE = 1024


def context_key(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def to_stored(packets: torch.Tensor, dtype: str) -> np.ndarray:
    packets = packets.detach().float().cpu()
    if dtype == "float32":
        return packets.numpy()
    return packets.to(torch.bfloat16).contiguous().view(torch.int16).numpy()


def from_stored(array: np.ndarray, dtype: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.array(array))
    if dtype == "float32":
        return tensor
    return tensor.view(torch.bfloat16).float()


def round_to_stored(packets: torch.Tensor, dtype: str) -> torch.Tensor:
    """Return 32-bit floats that hold exactly the values a stored packet would hold."""
    if dtype == "float32":
        return packets
    return packets.to(torch.bfloat16).float()


class PacketWriter:
    """Write the packets produced by one process into shards."""

    def __init__(self, directory: str | Path, rank: int, dtype: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.dtype = dtype
        self.index: dict[str, list] = {}
        self._pending: list[np.ndarray] = []
        self._keys: list[str] = []
        self._shards = 0

    def add(self, keys: Sequence[str], packets: torch.Tensor) -> None:
        self._keys.extend(keys)
        self._pending.extend(to_stored(packets, self.dtype))
        if len(self._pending) >= SHARD_SIZE:
            self._flush()

    def _flush(self) -> None:
        name = f"packets-rank{self.rank}-{self._shards:05d}.npy"
        np.save(self.directory / name, np.stack(self._pending))
        for row, key in enumerate(self._keys):
            self.index[key] = [name, row]
        self._pending, self._keys = [], []
        self._shards += 1

    def close(self) -> None:
        if self._pending:
            self._flush()
        (self.directory / f"index-rank{self.rank}.json").write_text(json.dumps(self.index))


class PacketCache:
    """Read stored packets by context. Shards are memory-mapped on first use."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.dtype = json.loads((self.directory / "meta.json").read_text())["dtype"]
        self.index: dict[str, list] = {}
        for path in sorted(self.directory.glob("index-rank*.json")):
            self.index.update(json.loads(path.read_text()))
        self._shards: dict[str, np.ndarray] = {}

    def _packet(self, context: str) -> torch.Tensor:
        name, row = self.index[context_key(context)]
        if name not in self._shards:
            self._shards[name] = np.load(self.directory / name, mmap_mode="r")
        return from_stored(self._shards[name][row], self.dtype)

    def __call__(self, examples: Sequence[Example]) -> torch.Tensor:
        return torch.stack([self._packet(example.context) for example in examples])
