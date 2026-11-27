"""Element-wise replacements for normalization layers.

Each layer computes ``gamma * f(x; alpha) + beta`` for every element on its
own, with no statistics over the batch or the features:

* ``DynamicTanh``: ``f = tanh(alpha * x)`` (DyT; Zhu et al., "Transformers
  without Normalization", 2025).
* ``DynamicSoftsign``: ``f = softsign(alpha * x) = alpha * x / (1 + |alpha * x|)``,
  DyT with softsign in place of tanh: the same slope ``alpha`` at 0 and limits
  of +-1, but reached polynomially (``1 - 1 / |alpha * x|``) rather than
  exponentially, so large inputs are squashed less.
* ``DynamicScale``: ``f = alpha * x``, with ``alpha`` starting at ``1 / sqrt(d)``
  (``d_model`` in the Transformer, the fan-in of the conv before the norm in a
  ResNet).

``gamma`` and ``beta`` are per channel and named ``weight`` and ``bias``, as in
BatchNorm and LayerNorm, so ``nstudy/norm_io.py`` can take them out to show
``f`` itself. ``alpha`` is one learned number per layer, or one per channel
with ``per_channel_alpha=True``; ``learn_alpha=False`` keeps it fixed at its
initial value. ``bias=False`` drops ``beta``. ``hadamard=True``
mixes the channels with a fixed rotation before ``gamma``: ``gamma * R f + beta``.

The channels are on axis 1 of 4-D inputs (images, as BatchNorm2d) and on the
last axis otherwise (token features, as LayerNorm), unless ``channel_dim`` says
otherwise.

A new transformation is a subclass that defines ``squash``::

    class DynamicErf(ElementwiseNorm):
        def squash(self, x, alpha):
            return torch.erf(alpha * x)

and a config file in ``config/norm/`` that names it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def hadamard_rotation(n: int, generator: torch.Generator | None = None) -> Tensor:
    """A randomized Hadamard rotation: ``H_n @ diag(signs) / sqrt(n)``.

    ``H_n`` is the Sylvester Hadamard matrix (entries +-1, ``n`` a power of 2);
    the random signs flip its columns, so every layer gets its own rotation.
    The result is orthogonal: ``R @ R.T`` is the identity.
    """
    if n < 1 or n & (n - 1):
        raise ValueError(f"a Hadamard rotation needs a power-of-2 size, got {n}")
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < n:
        matrix = torch.cat([torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)], 0)
    signs = torch.randint(0, 2, (n,), generator=generator).float() * 2 - 1
    return matrix * signs / n**0.5


class ElementwiseNorm(nn.Module):
    """``gamma * squash(x, alpha) + beta``, with ``squash`` defined by subclasses.

    Args:
        num_features: channels (ResNet) or model dimension (Transformer).
        alpha_init: initial value of ``alpha``.
        per_channel_alpha: one ``alpha`` per channel instead of one per layer.
        learn_alpha: train ``alpha``. False keeps it at ``alpha_init`` as a
            buffer (still named ``alpha``, saved in checkpoints, no gradient
            and no weight decay), so only ``gamma`` and ``beta`` are learned.
        channel_dim: axis of the channels; None: 1 for 4-D inputs, else the last.
        bias: learn the shift ``beta``. Without it, a ResNet's convs (built
            without biases, as BatchNorm's ``beta`` stands in for them) have no
            bias at all.
        dropout: probability of zeroing each element of the output (after
            ``gamma`` and ``beta``) in training, as regularization; 0 for none.
        dropout_channels: zero whole channels of an image instead of single
            elements (``nn.Dropout2d``), for conv feature maps (4-D inputs) only.
        hadamard: mix the channels with a fixed randomized Hadamard rotation
            ``R`` (see ``hadamard_rotation``) between ``squash`` and ``gamma``:
            ``gamma * R squash(x, alpha) + beta``. ``R`` is drawn once, from the
            random state at construction (the run's seed), and kept in the
            checkpoint as the buffer ``rotation``; it is not trained.
            ``num_features`` must be a power of 2.
    """

    def __init__(
        self,
        num_features: int,
        alpha_init: float = 0.5,
        per_channel_alpha: bool = False,
        learn_alpha: bool = True,
        channel_dim: int | None = None,
        bias: bool = True,
        dropout: float = 0.0,
        dropout_channels: bool = False,
        hadamard: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.channel_dim = channel_dim
        alpha = torch.full((num_features if per_channel_alpha else 1,), float(alpha_init))
        self.learn_alpha = learn_alpha
        if learn_alpha:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features)) if bias else None
        # No parameters or buffers: checkpoints load with any dropout.
        self.dropout_channels = dropout_channels
        if dropout == 0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout2d(dropout) if dropout_channels else nn.Dropout(dropout)
        self.hadamard = hadamard
        if hadamard:
            self.register_buffer("rotation", hadamard_rotation(num_features))

    def squash(self, x: Tensor, alpha: Tensor) -> Tensor:
        raise NotImplementedError

    def channel_axis(self, x: Tensor) -> int:
        if self.channel_dim is not None:
            return self.channel_dim
        return 1 if x.dim() == 4 else -1

    def per_channel(self, parameter: Tensor, x: Tensor) -> Tensor:
        """``parameter`` shaped to broadcast along the channel axis of ``x``."""
        axis = self.channel_axis(x) % x.dim()
        shape = [1] * x.dim()
        shape[axis] = parameter.numel()
        return parameter.view(shape)

    def forward(self, x: Tensor) -> Tensor:
        out = self.squash(x, self.per_channel(self.alpha, x))
        if self.hadamard:
            axis = self.channel_axis(x)
            out = (out.movedim(axis, -1) @ self.rotation.T).movedim(-1, axis)
        out = out * self.per_channel(self.weight, x)
        if self.bias is not None:
            out = out + self.per_channel(self.bias, x)
        if self.dropout_channels and out.dim() != 4:
            raise ValueError("dropout_channels needs (batch, channels, height, width) inputs")
        return self.dropout(out)

    def extra_repr(self) -> str:
        return (f"{self.num_features}, alpha={self.alpha.numel()} value(s)"
                f"{'' if self.learn_alpha else ' (fixed)'}, "
                f"bias={self.bias is not None}")


class DynamicTanh(ElementwiseNorm):
    """``gamma * tanh(alpha * x) + beta`` (DyT; ``alpha`` starts at 0.5)."""

    def squash(self, x: Tensor, alpha: Tensor) -> Tensor:
        return torch.tanh(alpha * x)


class DynamicSoftsign(ElementwiseNorm):
    """``gamma * softsign(alpha * x) + beta`` (``alpha`` starts at 0.5, as DyT)."""

    def squash(self, x: Tensor, alpha: Tensor) -> Tensor:
        return nn.functional.softsign(alpha * x)


class DynamicScale(ElementwiseNorm):
    """``gamma * alpha * x + beta``: a learned scale, with ``alpha`` starting at ``1 / sqrt(d)``.

    So the layer starts as ``x / sqrt(d)``. ``alpha`` multiplies rather than
    divides: the gradient of ``x / alpha`` with respect to ``alpha`` is
    ``-x / alpha ** 2``, tiny for ``alpha`` near ``sqrt(d)``, so it hardly
    learned; that of ``alpha * x`` is ``x``.

    ``d`` is the fan-in of what produces the input: ``d_model`` in the
    Transformer, where ``d = num_features``; ``in_channels * k * k`` of the conv
    before the norm in a ResNet, which the ResNet sets with ``set_fan_in`` once
    the layer exists (the layer itself only knows its own channels). A number
    given as ``alpha_init`` is used as is instead.

    Only ``gamma * alpha`` matters for the output; the two differ in how they
    are learned. Both get the optimizer's weight decay, like every parameter.
    ``learn_alpha=False`` removes ``alpha`` (it is fixed at 1) and starts
    ``gamma`` at ``1 / sqrt(d)`` (or ``alpha_init``) instead: the layer is
    ``gamma * x + beta``, starting as ``x / sqrt(d)``. A fixed ``alpha`` of
    ``1 / sqrt(d)`` with ``gamma`` from 1 would learn the scale ``1 / alpha**2``
    times slower, as the gradient of ``gamma`` carries a factor ``alpha``.
    """

    def __init__(
        self, num_features: int, alpha_init: float | None = None, learn_alpha: bool = True, **kwargs
    ) -> None:
        self.automatic = alpha_init is None
        scale = num_features**-0.5 if alpha_init is None else alpha_init
        super().__init__(num_features, alpha_init=scale if learn_alpha else 1.0,
                         learn_alpha=learn_alpha, **kwargs)
        if not learn_alpha:
            with torch.no_grad():
                self.weight.fill_(scale)

    def initial_scale(self) -> Tensor:
        """What starts at ``1 / sqrt(d)``: ``alpha``, or ``gamma`` without a learned ``alpha``."""
        return self.alpha if self.learn_alpha else self.weight

    def set_fan_in(self, fan_in: int) -> None:
        """Start the scale at ``1 / sqrt(fan_in)``, unless ``alpha_init`` was given."""
        if self.automatic:
            with torch.no_grad():
                self.initial_scale().fill_(fan_in**-0.5)

    def squash(self, x: Tensor, alpha: Tensor) -> Tensor:
        return alpha * x

class LayerNorm2d(nn.GroupNorm):
    """LayerNorm for conv feature maps: each image normalized over (C, H, W).

    The LayerNorm of Ba et al. (2016) as used for CNNs in Wu & He, "Group
    Normalization" (2018): ``GroupNorm`` with one group, so a per-channel
    ``weight`` and ``bias`` as in BatchNorm. ``nn.LayerNorm(C)`` would normalize
    only the last axis (width) of an (N, C, H, W) input.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__(1, num_features, eps=eps, affine=affine)
