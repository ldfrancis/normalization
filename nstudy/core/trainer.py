"""Training loop and the learning-rate schedules of the two papers.

Training is counted in steps (optimizer updates), not epochs, because both
He et al. (2016) and Vaswani et al. (2017) specify their schedules that way.

The trainer runs in one process, or on several GPUs with DistributedDataParallel
when started by ``torchrun`` (see ``nstudy.core.distributed``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Protocol, Sequence

import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader

from nstudy.core import distributed
from nstudy.core.evaluator import Task, evaluate

if TYPE_CHECKING:
    from nstudy.core.probes import ProbeRecorder

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learning-rate schedules
# ---------------------------------------------------------------------------


def step_factor(
    step: int,
    milestones: Sequence[int],
    gamma: float = 0.1,
    warmup_steps: int = 0,
    warmup_factor: float = 0.1,
) -> float:
    """Multiplier of the base learning rate for update number ``step`` (from 0).

    The rate is multiplied by ``gamma`` at every milestone. During the first
    ``warmup_steps`` updates it is ``warmup_factor`` times the base rate, which
    He et al. use for ResNet-110 (0.01 for about 400 iterations, then 0.1).
    """
    if step < warmup_steps:
        return warmup_factor
    return gamma ** sum(step >= milestone for milestone in milestones)


def step_schedule(
    optimizer: Optimizer,
    milestones: Sequence[int],
    gamma: float = 0.1,
    warmup_steps: int = 0,
    warmup_factor: float = 0.1,
) -> LambdaLR:
    """Piecewise-constant schedule of the ResNet paper; call ``step()`` once per update."""
    milestones = list(milestones)
    return LambdaLR(
        optimizer, lambda step: step_factor(step, milestones, gamma, warmup_steps, warmup_factor)
    )


def noam_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """Learning rate of Vaswani et al., Equation 3, for update number ``step`` (from 1).

    ``d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)``: a linear
    increase for ``warmup_steps`` updates, then decay with the inverse square
    root of the step.
    """
    return d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)


def noam_schedule(optimizer: Optimizer, d_model: int, warmup_steps: int = 4000) -> LambdaLR:
    """Schedule of the Transformer paper. The optimizer's base rate must be 1."""
    return LambdaLR(optimizer, lambda step: noam_lr(step + 1, d_model, warmup_steps))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def pick_device() -> torch.device:
    """This process's GPU (``LOCAL_RANK`` under torchrun), else MPS, else the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    if distributed.is_distributed():
        return torch.device("cpu")  # the gloo backend cannot exchange MPS tensors
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cycle(loader: DataLoader) -> Iterator:
    """Yield batches forever, starting a new pass over ``loader`` whenever one ends.

    A ``DistributedSampler`` is told the pass number so each pass is shuffled anew.
    """
    passes = 0
    while True:
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(passes)
        yield from loader
        passes += 1


PRECISIONS = {"fp32": None, "bf16": torch.bfloat16}


class Logger(Protocol):
    """Receives every row written to ``metrics.jsonl`` (see ``nstudy.core.tracking``).

    A logger may also have ``log_histograms(step, histograms)``, which then
    receives the probes' histograms, and ``summary(values)``, which then
    receives the final metrics as ``final/<split>/<metric>``.
    """

    def log(self, row: Mapping[str, Any]) -> None: ...


def norm_of(tensors: Iterator[torch.Tensor]) -> float:
    """Euclidean norm of all the given tensors taken as one vector."""
    norms = [tensor.detach().float().norm() for tensor in tensors]
    return torch.stack(norms).norm().item() if norms else 0.0


def total_norm(tensors: list[torch.Tensor]) -> torch.Tensor:
    """``norm_of`` as a tensor on the tensors' device: no wait for the device.

    ``get_total_norm`` (PyTorch 2.6+, used by ``clip_grad_norm_``) takes all the
    norms in a few fused kernels, so this is cheap enough for every step.
    """
    tensors = [tensor.detach() for tensor in tensors]
    get_total_norm = getattr(torch.nn.utils, "get_total_norm", None)
    if get_total_norm is not None:
        return get_total_norm(tensors).float()
    return torch.linalg.vector_norm(torch.stack([tensor.float().norm() for tensor in tensors]))


class Trainer:
    """Train a model for a fixed number of optimizer updates.

    Args:
        model: the network to train.
        loaders: dataloaders by split name. ``"train"`` is required; every
            other split is held out and evaluated at the end of training.
        task: how a batch becomes a loss (see ``nstudy.core.evaluator``).
        optimizer: optimizer over ``model``'s parameters.
        scheduler: learning-rate schedule, stepped once per update.
        max_steps: total number of optimizer updates.
        output_dir: where ``metrics.jsonl`` and ``checkpoint.pt`` are written.
        accumulate: batches whose gradients are averaged into one update. Use
            it to reach a batch size that does not fit in memory at once.
        eval_every: evaluate ``eval_splits`` and save a checkpoint every this many updates.
        log_every: record the training metrics every this many updates:

            * ``loss``: mean training loss since the last record.
            * ``error`` (classification): percentage of wrong predictions on
              the (augmented) training batches since the last record.
            * ``lr``: learning rate of the update.
            * ``epoch``: passes over the training data so far.
            * ``throughput``: training targets per second over all processes
              (images, or non-padding target tokens), evaluation time excluded.
            * ``step_time``: seconds per update.
            * ``grad_norm``: norm of all gradients of the last update before
              the record (after averaging across processes; before weight
              decay and momentum, which the optimizer adds). One minibatch's
              gradient, so it is noisy.
            * ``grad_norm_mean``: the mean of that norm over every update since
              the last record: smoother, as it averages ``log_every`` minibatches.
            * ``<block>/<weight>_grad_norm`` and ``<block>/<weight>_grad_norm_mean``:
              the same for each weight the probes watch, from the training
              gradients, not the probe batch: the ``conv1`` weights of an
              early, a middle and a late ResNet block; in the Transformer, the
              attention ``out_proj`` and feed-forward ``linear2`` weights of
              an early, a middle and a late encoder and decoder layer.
            * ``param_norm``: norm of all parameters after the update.
            * ``max_memory_gb`` (CUDA): peak memory of rank 0's GPU since the
              last record.

            Evaluation rows also hold ``seconds``, the time the evaluation took.
        eval_splits: held-out splits evaluated every ``eval_every`` updates;
            ``"val"`` (if there is one) when None.
        device: where to train; the best available device when None.
        precision: ``"fp32"`` (as in the papers) or ``"bf16"``, which runs the
            forward pass in bfloat16 autocast. Evaluation always uses fp32.
        logger: also receives every row written to ``metrics.jsonl``, such as
            ``nstudy.core.tracking.WandbLogger``. Only rank 0 calls it.
        probes: measures chosen layers on a fixed batch of training examples
            on its own schedule (see ``nstudy.core.probes``), after the update,
            on rank 0. Statistics are written as ``"probe"`` rows of
            ``metrics.jsonl``; histograms go to ``histograms.jsonl`` and to the
            logger. Its time is left out of the training throughput.

    With several processes, ``model`` is wrapped in DistributedDataParallel
    (``self.model``; the plain network is ``self.module``), gradients are
    averaged across processes, and only rank 0 logs and writes files.
    BatchNorm normalizes with each process's own part of the batch.
    """

    def __init__(
        self,
        model: nn.Module,
        loaders: dict[str, DataLoader],
        task: Task,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        max_steps: int,
        output_dir: str | Path,
        accumulate: int = 1,
        eval_every: int = 1000,
        log_every: int = 100,
        eval_splits: Sequence[str] | None = None,
        device: torch.device | None = None,
        precision: str = "fp32",
        logger: Logger | None = None,
        probes: ProbeRecorder | None = None,
    ) -> None:
        if eval_splits is None:
            eval_splits = ["val"] if "val" in loaders else []
        for split in eval_splits:
            if split == "train" or split not in loaders:
                held_out = sorted(set(loaders) - {"train"})
                raise ValueError(f"eval_splits must be among the held-out splits {held_out}, "
                                 f"got {split!r}")
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
        self.device = device or pick_device()
        self.module = model.to(self.device)
        self.model: nn.Module = self.module
        if distributed.is_distributed():
            device_ids = [self.device] if self.device.type == "cuda" else None
            self.model = DistributedDataParallel(self.module, device_ids=device_ids)
        self.precision = precision
        self.loaders = loaders
        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_steps = max_steps
        self.accumulate = accumulate
        self.eval_every = eval_every
        self.log_every = log_every
        self.eval_splits = list(eval_splits)
        self.logger = logger
        self.probes = probes
        # This process's wrong predictions and predicted targets since the
        # last record of the training metrics.
        self.train_wrong = 0
        self.train_seen = 0
        # Gradient norms, kept by rank 0 (after DDP every process has the same
        # gradients): "" is all parameters, other keys the probes' weights.
        # Sums since the last record stay on the device until the record.
        self.watched: dict[str, list[torch.Tensor]] = {"": list(self.module.parameters())}
        if probes is not None:
            for probe in probes.probes:
                if probe.weight is not None:
                    self.watched[f"{probe.name}/{probe.weight_name}"] = [probe.weight]
        self.grad_norm_sums: dict[str, torch.Tensor] = {}
        self.grad_norm_last: dict[str, torch.Tensor] = {}
        self.grad_norm_count = 0
        self.output_dir = Path(output_dir)
        if distributed.is_main():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "checkpoint.pt"

    def save_checkpoint(self) -> None:
        """Rank 0 saves; every process must call this, as it syncs BatchNorm statistics."""
        distributed.broadcast_buffers(self.module)
        if not distributed.is_main():
            return
        torch.save(
            {
                "step": self.step,
                "model": self.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            self.checkpoint_path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore the model, optimizer, schedule and step count saved by ``save_checkpoint``."""
        state = torch.load(path, map_location=self.device)
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = state["step"]

    def record(self, split: str, metrics: dict[str, float]) -> None:
        """Log one line of metrics and append it to ``metrics.jsonl`` (rank 0 only)."""
        if not distributed.is_main():
            return
        row = {"step": self.step, "split": split, **metrics}
        log.info(" ".join(f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
                          for key, value in row.items()))
        with open(self.output_dir / "metrics.jsonl", "a") as file:
            file.write(json.dumps(row) + "\n")
        if self.logger is not None:
            self.logger.log(row)

    def record_histograms(self, histograms: dict[str, dict[str, list]]) -> None:
        """Append histograms to ``histograms.jsonl`` and give them to the logger (rank 0 only)."""
        if not distributed.is_main():
            return
        with open(self.output_dir / "histograms.jsonl", "a") as file:
            file.write(json.dumps({"step": self.step, "histograms": histograms}) + "\n")
        log_histograms = getattr(self.logger, "log_histograms", None)
        if log_histograms is not None:
            log_histograms(self.step, histograms)

    def autocast(self) -> contextlib.AbstractContextManager:
        dtype = PRECISIONS[self.precision]
        if dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=dtype)

    def train_step(self, batches: Iterator) -> float:
        """Do one optimizer update and return the training loss (averaged over processes)."""
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for index in range(self.accumulate):
            # DDP averages gradients across processes in backward; only the
            # last accumulated batch needs to.
            last = index == self.accumulate - 1
            sync = (
                self.model.no_sync()
                if isinstance(self.model, DistributedDataParallel) and not last
                else contextlib.nullcontext()
            )
            with sync:
                with self.autocast():
                    logits, targets = self.task.forward(self.model, next(batches), self.device)
                    loss = self.task.training_loss(logits, targets) / self.accumulate
                loss.backward()
            total += loss.item()
            with torch.no_grad():
                counted = targets != self.task.ignore_index
                self.train_seen += int(counted.sum())
                if self.task.report_error:
                    self.train_wrong += int(((logits.argmax(dim=-1) != targets) & counted).sum())
        # Every process sees the same value, so they all stop together if it is not finite.
        total = distributed.all_reduce_sum([total], self.device)[0] / distributed.world_size()
        if not math.isfinite(total):
            raise FloatingPointError(f"training loss is {total} at step {self.step}")
        self.track_gradient_norms()
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return total

    def track_gradient_norms(self) -> None:
        """Add this update's gradient norms to the sums since the last record (rank 0)."""
        if not distributed.is_main():
            return
        for key, parameters in self.watched.items():
            grads = [p.grad for p in parameters if p.grad is not None]
            if not grads:
                continue
            norm = total_norm(grads)
            self.grad_norm_last[key] = norm
            self.grad_norm_sums[key] = self.grad_norm_sums.get(key, 0) + norm
        self.grad_norm_count += 1

    def gradient_norm_metrics(self) -> dict[str, float]:
        """``grad_norm`` and ``grad_norm_mean`` of every watched tensor; resets the sums."""
        metrics: dict[str, float] = {}
        for key in self.watched:
            name = f"{key}_grad_norm" if key else "grad_norm"
            last, total = self.grad_norm_last.get(key), self.grad_norm_sums.get(key)
            metrics[name] = math.nan if last is None else last.item()
            metrics[f"{name}_mean"] = (math.nan if total is None
                                       else total.item() / self.grad_norm_count)
        self.grad_norm_sums, self.grad_norm_last, self.grad_norm_count = {}, {}, 0
        return metrics

    def run_probes(self) -> None:
        """Measure the probes on their fixed batch if this step is on their schedule."""
        if self.probes is None or not distributed.is_main():
            return
        want_stats, want_histograms = self.probes.due(self.step)
        if not (want_stats or want_histograms):
            return
        statistics, histograms = self.probes.run(
            self.task, self.device, want_stats, want_histograms, self.autocast
        )
        if want_stats:
            self.record("probe", statistics)
        if want_histograms:
            self.record_histograms(histograms)

    def epoch(self) -> float | None:
        """Passes over the training data so far, if the loader has a length."""
        try:
            return self.step * self.accumulate / len(self.loaders["train"])
        except TypeError:
            return None

    def evaluate_and_record(self, split: str) -> dict[str, float]:
        start = time.perf_counter()
        metrics = evaluate(self.module, self.loaders[split], self.task, self.device)
        self.record(split, {**metrics, "seconds": time.perf_counter() - start})
        return metrics

    def training_metrics(self, losses: list[float], learning_rate: float,
                         seconds: float) -> dict[str, float]:
        """The row of training metrics since the last one (see the class docstring)."""
        wrong, seen = distributed.all_reduce_sum([self.train_wrong, self.train_seen], self.device)
        self.train_wrong = self.train_seen = 0
        metrics = {"loss": sum(losses) / len(losses)}
        if self.task.report_error:
            metrics["error"] = 100.0 * wrong / seen
        metrics["lr"] = learning_rate
        epoch = self.epoch()
        if epoch is not None:
            metrics["epoch"] = epoch
        metrics["throughput"] = seen / seconds
        metrics["step_time"] = seconds / len(losses)
        if distributed.is_main():
            metrics.update(self.gradient_norm_metrics())
        metrics["param_norm"] = norm_of(self.module.parameters())
        if self.device.type == "cuda":
            metrics["max_memory_gb"] = torch.cuda.max_memory_allocated(self.device) / 2**30
            torch.cuda.reset_peak_memory_stats(self.device)
        return metrics

    def fit(self) -> dict[str, dict[str, float]]:
        """Train until ``max_steps`` and return the final metrics of every held-out split."""
        self.model.train()
        batches = cycle(self.loaders["train"])
        recent: list[float] = []
        window_start = time.perf_counter()
        while self.step < self.max_steps:
            learning_rate = self.optimizer.param_groups[0]["lr"]
            recent.append(self.train_step(batches))
            paused = time.perf_counter()
            self.run_probes()
            window_start += time.perf_counter() - paused
            if self.step % self.log_every == 0:
                seconds = time.perf_counter() - window_start
                self.record("train", self.training_metrics(recent, learning_rate, seconds))
                recent.clear()
                window_start = time.perf_counter()
            if self.step % self.eval_every == 0 and self.step < self.max_steps:
                paused = time.perf_counter()
                self.save_checkpoint()
                for split in self.eval_splits:
                    self.evaluate_and_record(split)
                # Leave checkpointing and evaluation out of the training throughput.
                window_start += time.perf_counter() - paused

        self.save_checkpoint()
        # The final model on every held-out split, including those (such as the
        # test set) not evaluated during training.
        results = {
            split: self.evaluate_and_record(split) for split in self.loaders if split != "train"
        }
        self.record_final(results)
        return results

    def record_final(self, results: dict[str, dict[str, float]]) -> None:
        """Write the final metrics to ``final_metrics.json`` and the logger's summary (rank 0)."""
        if not distributed.is_main():
            return
        final = {"step": self.step, **results}
        (self.output_dir / "final_metrics.json").write_text(json.dumps(final, indent=2) + "\n")
        log.info("final metrics after %d updates: %s", self.step, json.dumps(results))
        summary = getattr(self.logger, "summary", None)
        if summary is not None:
            summary({f"final/{split}/{key}": value
                     for split, metrics in results.items() for key, value in metrics.items()})
