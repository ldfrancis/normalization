"""What normalization layers of a trained model compute, before their scale and shift.

A normalization layer computes ``y = gamma * x_hat + beta``, where ``x_hat`` is
its input ``x`` normalized (``(x - mean) / sqrt(var + eps)`` for BatchNorm and
LayerNorm, over different axes). This module records, for many elements of
the input, the pairs ``(x, x_hat)``, and also ``y``, so ``x_hat`` against ``x``
shows the function the layer applies before its learned affine transform.

The layers are the probes' (``nstudy.core.probes.model_probes``): an early, a
middle and a late block; for ResNets the norm whose output feeds the ReLU after
``conv1``, for the Transformer every norm of the encoder and decoder layers.

``x_hat`` comes from a copy of the layer whose scale is 1 and shift is 0, run
on the same input, so any layer with a ``weight`` (scale) and ``bias`` (shift)
works, whatever its normalization (BatchNorm, LayerNorm, GroupNorm, RMSNorm),
including the element-wise layers of ``nstudy.core.norms`` (DyT: ``x_hat =
tanh(alpha * x)``). Without a norm (``norm=none``), ``x_hat = y = x``.

Two modes:

* ``"eval"``: the model as used after training. BatchNorm normalizes with its
  running statistics, so for each channel ``x_hat`` is a straight line of ``x``.
  LayerNorm normalizes each token by its own statistics, so the points of one
  channel do not lie on one line.
* ``"train"``: normalization layers with running statistics use the statistics
  of each batch instead, as during training. Dropout stays off in both modes.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from nstudy.core.evaluator import Task
from nstudy.core.norms import ElementwiseNorm
from nstudy.core.probes import Probe

MODES = ("eval", "train")
FIELDS = ("x", "x_hat", "y", "channel")


def unit_affine(module: nn.Module) -> nn.Module:
    """A copy of ``module`` with scale 1 and shift 0: its output is ``x_hat``."""
    unit = copy.deepcopy(module)
    with torch.no_grad():
        weight = getattr(unit, "weight", None)
        bias = getattr(unit, "bias", None)
        if isinstance(weight, Tensor):
            weight.fill_(1.0)
        if isinstance(bias, Tensor):
            bias.zero_()
    return unit


def channel_dim(module: nn.Module, x: Tensor) -> int:
    """Axis of the channels of ``x``: the layer's own if it says (``nstudy.core.norms``),
    1 for BatchNorm, InstanceNorm and GroupNorm, else the last."""
    if isinstance(module, ElementwiseNorm):
        return module.channel_axis(x)
    if isinstance(module, (nn.modules.batchnorm._NormBase, nn.GroupNorm)):
        return 1
    return 1 if isinstance(module, nn.Identity) and x.dim() == 4 else -1


def has_running_stats(module: nn.Module) -> bool:
    return getattr(module, "track_running_stats", False) and getattr(module, "running_mean", None) is not None


@dataclass
class LayerRecord:
    """The recorded points of one layer: equal-length 1-D tensors."""

    name: str
    norm: str
    position: str
    points: dict[str, list[Tensor]] = field(default_factory=lambda: {key: [] for key in FIELDS})

    def tensors(self) -> dict[str, Tensor]:
        return {key: torch.cat(values) if values else torch.empty(0) for key, values in self.points.items()}


def positions(count: int) -> list[str]:
    """Names for the watched blocks: early, mid, late (or block numbers if more)."""
    return ["early", "mid", "late"][:count] if count <= 3 else [str(i) for i in range(count)]


class NormIORecorder:
    """Record ``(x, x_hat, y)`` of the probes' normalization layers on many examples.

    Args:
        model: the trained network (its parameters and buffers are not changed).
        probes: the layers to watch.
        points_per_batch: elements kept per layer and batch, chosen at random
            (an early ResNet layer has millions of elements per batch).
        mode: ``"eval"`` or ``"train"`` (see the module docstring).
        seed: seed of the random choice of elements.
    """

    def __init__(
        self,
        model: nn.Module,
        probes: list[Probe],
        points_per_batch: int,
        mode: str = "eval",
        seed: int = 0,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.model = model
        self.probes = probes
        self.points_per_batch = points_per_batch
        self.mode = mode
        self.generator = torch.Generator().manual_seed(seed)
        self.pad_id = getattr(model, "pad_id", None)
        self.masks: dict[str, Tensor] = {}
        self.units = {probe.name: unit_affine(probe.module) for probe in probes}
        self.records: dict[str, LayerRecord] = {}
        self.examples = 0
        self.handles = []
        for probe in probes:
            self.handles.append(probe.module.register_forward_hook(self.hook(probe)))
        if any(probe.side for probe in probes):
            self.handles.append(model.register_forward_pre_hook(self.keep_masks))

    def keep_masks(self, module: nn.Module, inputs: tuple) -> None:
        src, tgt = inputs[:2]
        self.masks = {"encoder": src != self.pad_id, "decoder": tgt != self.pad_id}

    def hook(self, probe: Probe):
        def record(module: nn.Module, inputs: tuple, output: Tensor) -> None:
            x = inputs[0].detach()
            unit = self.units[probe.name]
            unit.train(module.training)
            buffers = [buffer.clone() for buffer in unit.buffers()]
            x_hat = unit(x).detach()
            with torch.no_grad():  # batch statistics must not drift the copy's buffers
                for buffer, saved in zip(unit.buffers(), buffers):
                    buffer.copy_(saved)
            self.keep(probe, module, x, x_hat, output.detach())

        return record

    def keep(self, probe: Probe, module: nn.Module, x: Tensor, x_hat: Tensor, y: Tensor) -> None:
        dim = channel_dim(module, x)
        rows = [t.movedim(dim, -1) for t in (x, x_hat, y)]  # channels last
        if probe.side:  # Transformer: (batch, tokens, d_model); drop padded tokens
            mask = self.masks[probe.side]
            rows = [t[mask] for t in rows]
        channels = rows[0].shape[-1]
        flat = [t.reshape(-1).float().cpu() for t in rows]
        count = flat[0].numel()
        chosen = torch.randperm(count, generator=self.generator)[: self.points_per_batch]
        record = self.records[probe.name]
        for key, values in zip(("x", "x_hat", "y"), flat):
            record.points[key].append(values[chosen])
        record.points["channel"].append(chosen % channels)

    @torch.no_grad()
    def run(self, batches: Iterable[Any], task: Task, device: torch.device, examples: int) -> int:
        """Pass batches through the model until at least ``examples`` examples were seen.

        Returns the number of examples used (whole batches, so possibly more).
        """
        was_training = self.model.training
        state = {key: value.clone() for key, value in self.model.state_dict().items()}
        names = positions(len({self.block_of(p) for p in self.probes}))
        blocks = sorted({self.block_of(p) for p in self.probes}, key=self.block_order)
        position_of = dict(zip(blocks, names))
        self.records = {
            p.name: LayerRecord(p.name, type(p.module).__name__, position_of[self.block_of(p)])
            for p in self.probes
        }
        try:
            self.model.eval()
            if self.mode == "train":
                for module in self.model.modules():
                    if has_running_stats(module):
                        module.train()
            self.examples = 0
            for batch in batches:
                task.forward(self.model, batch, device)
                self.examples += batch_size(batch)
                if self.examples >= examples:
                    break
        finally:
            self.model.load_state_dict(state)
            self.model.train(was_training)
        return self.examples

    @staticmethod
    def block_of(probe: Probe) -> str:
        """``"layer2.4"`` -> ``"layer2.4"``; ``"encoder.3.norm2"`` -> ``"3"`` (the layer index)."""
        return probe.name.split(".")[1] if probe.side else probe.name

    @staticmethod
    def block_order(block: str) -> tuple:
        return tuple(int(part) if part.isdigit() else part for part in block.replace("layer", "").split("."))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def batch_size(batch: Any) -> int:
    """Number of examples (images or sentence pairs) in a batch."""
    if isinstance(batch, dict):
        return len(batch["src"])
    return len(batch[0])


def per_batch_quota(max_points: int, examples: int, batch_examples: int) -> int:
    """Elements to keep per batch so that about ``max_points`` are kept over ``examples``."""
    batches = max(1, math.ceil(examples / max(1, batch_examples)))
    return max(1, math.ceil(max_points / batches))


# ---------------------------------------------------------------------------
# Saving and plotting
# ---------------------------------------------------------------------------


def save(records: dict[str, LayerRecord], path, metadata: dict[str, Any]) -> None:
    """Save every layer's points in one ``.pt`` file (``torch.load`` reads it back)."""
    torch.save(
        {
            "metadata": metadata,
            "layers": {
                name: {"norm": r.norm, "position": r.position, **r.tensors()}
                for name, r in records.items()
            },
        },
        path,
    )


def row_name(name: str) -> str:
    """The row of a layer in the figure: its kind without the block index."""
    parts = name.split(".")
    if parts[0] in ("encoder", "decoder"):
        return f"{parts[0]} {parts[2]}"
    return "norm after conv1"


def plot(layers: dict[str, dict[str, Any]], path, channels: int = 8, title: str = "") -> None:
    """Scatter ``x_hat`` against ``x`` for each layer: rows by kind, columns early/mid/late.

    The points of ``channels`` channels are coloured (one colour each); the
    rest are grey.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(dict.fromkeys(row_name(name) for name in layers))
    columns = list(dict.fromkeys(layer["position"] for layer in layers.values()))
    figure, axes = plt.subplots(
        len(rows), len(columns), figsize=(4 * len(columns), 3.6 * len(rows)), squeeze=False
    )
    colours = plt.get_cmap("tab10")
    for name, layer in layers.items():
        axis = axes[rows.index(row_name(name))][columns.index(layer["position"])]
        x, x_hat, channel = (torch.as_tensor(layer[k]) for k in ("x", "x_hat", "channel"))
        shown = torch.unique(channel)[:channels]
        rest = ~torch.isin(channel, shown)
        axis.scatter(x[rest], x_hat[rest], s=1, c="0.75", alpha=0.3, rasterized=True, linewidths=0)
        for index, value in enumerate(shown):
            mine = channel == value
            axis.scatter(x[mine], x_hat[mine], s=2, color=colours(index % 10), alpha=0.6,
                         rasterized=True, linewidths=0)
        axis.set_title(f"{name} ({layer['position']})", fontsize=10)
        axis.set_xlabel("input x")
        axis.set_ylabel("x_hat (before scale and shift)")
        axis.grid(alpha=0.3)
    for row in axes:
        for axis in row:
            if not axis.has_data():
                axis.set_visible(False)
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
