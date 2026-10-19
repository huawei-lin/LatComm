"""Multi-GPU data parallelism: one process per GPU, launched with torchrun."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import os
from typing import Any

import torch
import torch.distributed as dist

# Gradients are reduced in buckets of about this many bytes.
BUCKET_BYTES = 25 * 1024**2


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda", torch.cuda.current_device())

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def all_gather(self, value: Any) -> list[Any]:
        """Collect one Python object from every process."""
        if self.world_size == 1:
            return [value]
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def sum(self, value: int) -> int:
        if self.world_size == 1:
            return value
        tensor = torch.tensor(value, dtype=torch.long, device=self.device)
        dist.all_reduce(tensor)
        return int(tensor)

    def sum_gradients(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Replace each gradient by its sum over all processes."""
        if self.world_size == 1:
            return
        bucket: list[torch.nn.Parameter] = []
        size = 0
        for parameter in parameters:
            bucket.append(parameter)
            size += parameter.numel() * parameter.element_size()
            if size >= BUCKET_BYTES:
                _reduce(bucket)
                bucket, size = [], 0
        if bucket:
            _reduce(bucket)

    def close(self) -> None:
        if self.world_size > 1:
            dist.destroy_process_group()


def _reduce(parameters: list[torch.nn.Parameter]) -> None:
    flat = torch.cat([parameter.grad.flatten() for parameter in parameters])
    dist.all_reduce(flat)
    offset = 0
    for parameter in parameters:
        parameter.grad.copy_(flat[offset : offset + parameter.numel()].view_as(parameter))
        offset += parameter.numel()


def initialize_distributed() -> DistributedContext:
    """Join the process group created by torchrun, or run as a single process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return DistributedContext()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    return DistributedContext(rank=int(os.environ["RANK"]), world_size=world_size)
