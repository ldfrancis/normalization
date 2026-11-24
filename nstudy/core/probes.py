"""Statistics of chosen layers during training.

Probes watch an early, a middle and a late block of the network:

* ResNet: the input of the ReLU after ``conv1`` in the block (the output of
  ``bn1``, or of whatever normalization layer replaces it), and the weights of
  ``conv1``, the layer whose output that normalization layer receives.
* Transformer: the output of every normalization layer of the layer
  (``norm1`` and ``norm2`` in the encoder; ``norm1``, ``norm2`` and ``norm3``
  in the decoder), separately for the encoder and the decoder, and the weight
  of the last linear of the sub-layer each norm wraps: ``self_attn.out_proj``
  (``norm1``), ``cross_attn.out_proj`` (decoder ``norm2``) and the
  feed-forward ``linear2`` (encoder ``norm2``, decoder ``norm3``). With
  post-norm (the paper) that sub-layer's output plus the residual is the
  norm's input; with pre-norm the norm's output is the sub-layer's input.

The values come from a separate pass on one fixed batch of training examples
(the ``"probe"`` loader of ``nstudy.core.data``: no augmentation, the same
examples every time), so changes between steps come from training and not
from different data. The pass runs in train mode, as training does, with the
same dropout masks every time; it leaves the model, its gradients and its
BatchNorm running statistics as they were. ``ProbeRecorder.run`` returns:

* statistics: for each watched tensor, its L2 norm, its root mean square
  (RMS) and its 15th, 50th and 85th percentiles (over all its elements), and
  the L2 norm and RMS of its gradient (of the training loss on the probe
  batch). The RMS, ``sqrt(mean(x ** 2)) = ||x||_2 / sqrt(n)``, is the typical
  size of one element, so it compares across layers of different sizes and
  across runs whose probe batches differ in size (different numbers of GPUs);
  the L2 norm is the size of the tensor as a whole. Both include the mean
  (``rms ** 2 = mean ** 2 + variance``); the percentiles show the spread;
* histograms: the distribution of each watched tensor's values.

Padded positions of Transformer batches are left out.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import torch
from torch import Tensor, nn

from nstudy.core.evaluator import Task
from nstudy.core.model import ResNet, Transformer

PERCENTILES = (15, 50, 85)


@dataclass(eq=False)
class Probe:
    """One watched place in the network.

    Args:
        name: what the logs call the place, such as ``"stage3.block9.layer54"``
            (ResNet-56: stage 3, its 9th block, whose ``conv1`` is the 54th
            weighted layer) or ``"decoder.layer1.norm2"``. Counted from 1.
        module: the module whose output is recorded.
        activation: what that output is called in the logs.
        weight: a parameter recorded alongside, or None.
        weight_name: what that parameter is called in the logs.
        side: ``"encoder"`` or ``"decoder"`` for Transformer outputs, whose
            padded positions are left out; None otherwise.
        block: index of the block (ResNet, over the whole network) or layer
            (Transformer, within its stack), from 0, to order probes by depth.
    """

    name: str
    module: nn.Module
    activation: str
    weight: nn.Parameter | None = None
    weight_name: str | None = None
    side: str | None = None
    block: int = 0


def default_blocks(count: int) -> list[int]:
    """The first, the middle and the last of ``count`` blocks."""
    return sorted({0, (count - 1) // 2, count - 1})


def pick(count: int, blocks: Sequence[int] | None) -> list[int]:
    """Block indices to watch; negative indices count from the end."""
    if blocks is None:
        return default_blocks(count)
    picked = []
    for index in blocks:
        if not -count <= index < count:
            raise IndexError(f"block {index} does not exist; there are {count} blocks")
        picked.append(index % count)
    return picked


def weighted_layers(model: ResNet) -> dict[nn.Module, int]:
    """The depth of each conv and linear, counted from 1 as in "ResNet-56".

    The stem conv is layer 1, the residual blocks' convs follow in order and
    the final linear is the last (56 in ResNet-56, 50 in ResNet-50). Projection
    shortcuts (option B) are not counted, as in the papers' depths.
    """
    modules = [
        module for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear)) and "downsample" not in name
    ]
    return {module: depth for depth, module in enumerate(modules, start=1)}


def resnet_probes(model: ResNet, blocks: Sequence[int] | None = None) -> list[Probe]:
    """The ReLU input after ``conv1`` and the ``conv1`` weights of chosen blocks.

    ``blocks`` indexes all residual blocks of the network in order, from 0
    (ResNet-56 has 27); by default the first, the middle and the last. Probes
    are named ``stage<s>.block<b>.layer<d>``: the stage, the block within it
    and the depth of its ``conv1`` (see ``weighted_layers``), all from 1. The
    model's own module names (``layer3.8``, as in torchvision) are unchanged.
    """
    # The model's stages are its modules layer1, layer2, ...
    named = [
        (f"stage{stage_name.removeprefix('layer')}.block{index + 1}", block)
        for stage_name, stage in model.named_children()
        if stage_name.startswith("layer")
        for index, block in enumerate(stage)
    ]
    depth = weighted_layers(model)
    probes = []
    for index in pick(len(named), blocks):
        name, block = named[index]
        probes.append(Probe(
            f"{name}.layer{depth[block.conv1]}", block.bn1, "relu_input",
            block.conv1.weight, "conv1_weight", block=index,
        ))
    return probes


# Each norm of a Transformer layer and the last linear of the sub-layer it wraps.
SUBLAYER_WEIGHTS = {
    "encoder": (("norm1", "self_attn.out_proj"), ("norm2", "linear2")),
    "decoder": (("norm1", "self_attn.out_proj"), ("norm2", "cross_attn.out_proj"),
                ("norm3", "linear2")),
}


def transformer_probes(model: Transformer, blocks: Sequence[int] | None = None) -> list[Probe]:
    """Every normalization layer in chosen encoder and decoder layers, with its sub-layer's weight.

    Each probe records the norm's output and the weight of the last linear of
    the sub-layer the norm wraps (``SUBLAYER_WEIGHTS``).

    ``blocks`` indexes the layers of the encoder and, separately, of the
    decoder; by default the first, the middle and the last.
    """
    probes = []
    for side in ("encoder", "decoder"):
        layers = getattr(model, side)
        for index in pick(len(layers), blocks):
            layer = layers[index]
            for norm_name, weight_name in SUBLAYER_WEIGHTS[side]:
                weight = layer.get_submodule(weight_name).weight
                probes.append(Probe(
                    f"{side}.layer{index + 1}.{norm_name}", getattr(layer, norm_name), "output",
                    weight, weight_name.replace(".", "_") + "_weight", side=side, block=index,
                ))
    return probes


def model_probes(model: nn.Module, blocks: Sequence[int] | None = None) -> list[Probe]:
    if isinstance(model, ResNet):
        return resnet_probes(model, blocks)
    if isinstance(model, Transformer):
        return transformer_probes(model, blocks)
    raise TypeError(f"no probes are defined for {type(model).__name__}")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def percentiles(tensor: Tensor, qs: Sequence[float]) -> list[float]:
    """Percentiles over all elements, by the nearest rank (no interpolation)."""
    flat = tensor.detach().float().flatten()
    if flat.device.type == "mps":  # kthvalue is not available on MPS
        flat = flat.cpu()
    count = flat.numel()
    return [flat.kthvalue(round(q / 100 * (count - 1)) + 1).values.item() for q in qs]


def rms(tensor: Tensor) -> float:
    """Root mean square of all elements: ``sqrt(mean(x ** 2))``."""
    return tensor.detach().float().pow(2).mean().sqrt().item()


def histogram(tensor: Tensor, bins: int) -> dict[str, list]:
    """Counts of the values in ``bins`` equal bins between their minimum and maximum.

    Counted on the CPU: ``bincount`` on a GPU adds atomically, which becomes very
    slow when most values share a bin (on MPS, seconds for a few million values),
    as trained activations often do. Copying a few million values is milliseconds.
    """
    flat = tensor.detach().float().flatten().cpu()
    low, high = flat.min().item(), flat.max().item()
    if high == low:
        low, high = low - 0.5, high + 0.5
    index = ((flat - low) / (high - low) * bins).long().clamp_(0, bins - 1)
    counts = torch.bincount(index, minlength=bins)
    edges = torch.linspace(low, high, bins + 1)
    return {"counts": counts.tolist(), "edges": edges.tolist()}


class ProbeRecorder:
    """Measure the probes on a fixed batch every so many training steps.

    Args:
        model: the network the probes belong to (not a DDP wrapper).
        probes: what to watch (see ``model_probes``).
        batch: the fixed batch of training examples (the ``"probe"`` loader's batch).
        histogram_every: record histograms every this many updates.
        stats_every: record norms and percentiles every this many updates.
        bins: number of histogram bins.
        seed: seed of the random numbers (dropout) used by every pass.
    """

    def __init__(
        self,
        model: nn.Module,
        probes: Sequence[Probe],
        batch: Any,
        histogram_every: int = 1000,
        stats_every: int = 1000,
        bins: int = 64,
        seed: int = 0,
    ) -> None:
        if not probes:
            raise ValueError("there is nothing to record: no probes")
        self.model = model
        self.probes = list(probes)
        self.batch = batch
        self.histogram_every = histogram_every
        self.stats_every = stats_every
        self.bins = bins
        self.seed = seed
        self.active = False
        self.activations: dict[str, Tensor] = {}
        self.masks: dict[str, Tensor] = {}
        for probe in self.probes:
            probe.module.register_forward_hook(self.hook(probe))
        if any(probe.side for probe in self.probes):
            model.register_forward_pre_hook(self.keep_masks)
        self.pad_id = getattr(model, "pad_id", None)

    def due(self, step: int) -> tuple[bool, bool]:
        """Whether update ``step`` (counted from 1) records statistics and histograms."""
        return step % self.stats_every == 0, step % self.histogram_every == 0

    def keep_masks(self, module: nn.Module, inputs: tuple) -> None:
        """Transformer ``forward(src, tgt)``: remember which positions are not padding."""
        if self.active:
            src, tgt = inputs[:2]
            self.masks = {"encoder": src != self.pad_id, "decoder": tgt != self.pad_id}

    def hook(self, probe: Probe):
        def record(module: nn.Module, inputs: tuple, output: Tensor) -> Tensor | None:
            if not self.active:
                return None
            self.activations[probe.name] = output
            # The next layer gets a copy, so an in-place ReLU changes the copy and
            # not the recorded tensor, whose gradient is that of the copy.
            return output.clone()

        return record

    @contextlib.contextmanager
    def preserved(self, device: torch.device) -> Iterator[None]:
        """Run a pass that leaves no trace: mode, buffers and random state are restored."""
        was_training = self.model.training
        buffers = [buffer.clone() for buffer in self.model.buffers()]
        devices = [device] if device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                if device.type == "mps":
                    mps_state = torch.mps.get_rng_state()
                torch.manual_seed(self.seed)  # the same dropout masks every pass
                try:
                    yield
                finally:
                    if device.type == "mps":
                        torch.mps.set_rng_state(mps_state)
        finally:
            with torch.no_grad():
                for buffer, saved in zip(self.model.buffers(), buffers):
                    buffer.copy_(saved)
            self.model.train(was_training)
            self.activations, self.masks = {}, {}
            self.active = False

    def select(self, probe: Probe, tensor: Tensor) -> Tensor:
        """The non-padding positions of a Transformer output, as (tokens, d_model)."""
        tensor = tensor.detach()
        return tensor[self.masks[probe.side]] if probe.side else tensor

    def run(
        self,
        task: Task,
        device: torch.device,
        stats: bool = True,
        histograms: bool = True,
        autocast: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext,
    ) -> tuple[dict[str, float], dict[str, dict[str, list]]]:
        """Pass the fixed batch through the model and measure the probes.

        Returns statistics keyed ``<probe>/<tensor>/<stat>`` (``l2``, ``rms``,
        ``p15``, ``p50``, ``p85``; ``l2`` and ``rms`` only for gradients) and histograms keyed
        ``<probe>/<tensor>``. Gradients are those of ``task``'s training loss on
        the batch, computed with ``torch.autograd.grad``, so the parameters'
        ``.grad`` (and DDP, which syncs them) are not touched.
        """
        with self.preserved(device):
            self.model.train()
            self.active = True
            with autocast():
                logits, targets = task.forward(self.model, self.batch, device)
                loss = task.training_loss(logits, targets)
            self.active = False

            tensors: list[tuple[str, Tensor, Tensor | None]] = []  # (key, values, for grad)
            for probe in self.probes:
                activation = self.activations[probe.name]
                tensors.append((f"{probe.name}/{probe.activation}", activation, probe))
                if probe.weight is not None:
                    tensors.append((f"{probe.name}/{probe.weight_name}", probe.weight, None))
            grads = torch.autograd.grad(
                loss, [tensor for _, tensor, _ in tensors], allow_unused=True
            )

            statistics: dict[str, float] = {}
            distributions: dict[str, dict[str, list]] = {}
            for (key, tensor, probe), grad in zip(tensors, grads):
                values = self.select(probe, tensor) if probe is not None else tensor.detach()
                if stats:
                    statistics[f"{key}/l2"] = values.float().norm().item()
                    statistics[f"{key}/rms"] = rms(values)
                    for q, value in zip(PERCENTILES, percentiles(values, PERCENTILES)):
                        statistics[f"{key}/p{q}"] = value
                    if grad is not None:
                        grad = self.select(probe, grad) if probe is not None else grad
                        statistics[f"{key}_grad/l2"] = grad.float().norm().item()
                        statistics[f"{key}_grad/rms"] = rms(grad)
                if histograms:
                    distributions[key] = histogram(values, self.bins)
        return statistics, distributions
