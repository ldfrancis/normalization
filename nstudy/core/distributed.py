"""Helpers for training on several GPUs with DistributedDataParallel (DDP).

Launch with ``torchrun``, which starts one process per GPU and sets
``WORLD_SIZE``, ``RANK`` and ``LOCAL_RANK``:

    torchrun --standalone --nproc_per_node=8 nstudy/main.py +experiment=imagenet_resnet

Without ``torchrun`` everything runs in one process, and every helper here
does nothing (rank 0 of a world of size 1).
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch import nn


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main() -> bool:
    """Only the first process logs and writes files."""
    return rank() == 0


def setup() -> None:
    """Join the process group when started by ``torchrun`` with more than one process."""
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1 or is_distributed():
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    else:
        dist.init_process_group("gloo")


def cleanup() -> None:
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


@contextlib.contextmanager
def main_process_first() -> Iterator[None]:
    """Let rank 0 run the block (download, tokenize, cache) before the others read the result."""
    if not is_main():
        barrier()
    yield
    if is_main():
        barrier()


def per_rank(total: int) -> int:
    """Each process's share of a global batch size, which must divide evenly."""
    world = world_size()
    if total % world:
        raise ValueError(f"batch size {total} does not divide among {world} processes")
    return total // world


def all_reduce_sum(values: list[float], device: torch.device) -> list[float]:
    """Sum numbers over all processes."""
    if not is_distributed():
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return tensor.tolist()


def gather_lists(items: list) -> list:
    """Concatenate every process's list, in rank order; every process gets the result."""
    if not is_distributed():
        return items
    parts: list[Any] = [None] * world_size()
    dist.all_gather_object(parts, items)
    return [item for part in parts for item in part]


def broadcast_object(value: Any) -> Any:
    """Give every process rank 0's ``value``."""
    if not is_distributed():
        return value
    box = [value]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def broadcast_buffers(module: nn.Module) -> None:
    """Copy rank 0's buffers (BatchNorm running statistics) to every process.

    Each rank updates its running statistics from its own images, so they
    drift apart between DDP's broadcasts. Syncing them before evaluating or
    saving makes every rank evaluate the model that is saved.
    """
    if is_distributed():
        for buffer in module.buffers():
            dist.broadcast(buffer, src=0)
