"""Element-wise replacements for normalization layers.

Each layer computes ``gamma * f(x; alpha) + beta`` for every element on its
own, with no statistics over the batch or the features:

* ``DynamicTanh``: ``f = tanh(alpha * x)`` (DyT; Zhu et al., "Transformers
  without Normalization", 2025).
* ``DynamicSoftsign``: ``f = x / (|alpha| + |x|)``, that is ``softsign(x / alpha)``.
  Its slope at 0 is ``1 / alpha`` (``alpha`` for tanh), so ``alpha = 2`` starts
  with the slope of DyT's ``alpha = 0.5``. ``|alpha|`` keeps the denominator
  positive if training pushes ``alpha`` below 0.
* ``DynamicScale``: ``f = alpha * x``, with ``alpha`` starting at ``1 / sqrt(d)``
  (``d_model`` in the Transformer, the fan-in of the conv before the norm in a
  ResNet).

``gamma`` and ``beta`` are per channel and named ``weight`` and ``bias``, as in
BatchNorm and LayerNorm, so ``nstudy/norm_io.py`` can take them out to show
``f`` itself. ``alpha`` is one learned number per layer, or one per channel
with ``per_channel_alpha=True``. ``bias=False`` drops ``beta``.

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


class ElementwiseNorm(nn.Module):
    """``gamma * squash(x, alpha) + beta``, with ``squash`` defined by subclasses.

    Args:
        num_features: channels (ResNet) or model dimension (Transformer).
        alpha_init: initial value of ``alpha``.
        per_channel_alpha: one ``alpha`` per channel instead of one per layer.
        channel_dim: axis of the channels; None: 1 for 4-D inputs, else the last.
        bias: learn the shift ``beta``. Without it, a ResNet's convs (built
            without biases, as BatchNorm's ``beta`` stands in for them) have no
            bias at all.
    """

    def __init__(
        self,
        num_features: int,
        alpha_init: float = 0.5,
        per_channel_alpha: bool = False,
        channel_dim: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.channel_dim = channel_dim
        self.alpha = nn.Parameter(torch.full((num_features if per_channel_alpha else 1,), float(alpha_init)))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features)) if bias else None

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
        out = self.squash(x, self.per_channel(self.alpha, x)) * self.per_channel(self.weight, x)
        return out if self.bias is None else out + self.per_channel(self.bias, x)

    def extra_repr(self) -> str:
        return (f"{self.num_features}, alpha={self.alpha.numel()} value(s), "
                f"bias={self.bias is not None}")


class DynamicTanh(ElementwiseNorm):
    """``gamma * tanh(alpha * x) + beta`` (DyT; ``alpha`` starts at 0.5)."""

    def squash(self, x: Tensor, alpha: Tensor) -> Tensor:
        return torch.tanh(alpha * x)


class DynamicSoftsign(ElementwiseNorm):
    """``gamma * x / (|alpha| + |x|) + beta`` (``alpha`` starts at 2: slope 0.5 at 0)."""

    def __init__(self, num_features: int, alpha_init: float = 2.0, **kwargs) -> None:
        super().__init__(num_features, alpha_init=alpha_init, **kwargs)

    def squash(self, x: Tensor, alpha: Tensor) -> Tensor:
        return x / (alpha.abs() + x.abs())


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
    """

    def __init__(self, num_features: int, alpha_init: float | None = None, **kwargs) -> None:
        self.automatic = alpha_init is None
        super().__init__(
            num_features,
            alpha_init=num_features**-0.5 if alpha_init is None else alpha_init,
            **kwargs,
        )

    def set_fan_in(self, fan_in: int) -> None:
        """Start ``alpha`` at ``1 / sqrt(fan_in)``, unless ``alpha_init`` was given."""
        if self.automatic:
            with torch.no_grad():
                self.alpha.fill_(fan_in**-0.5)

    def squash(self, x: Tensor, alpha: Tensor) -> Tensor:
        return alpha * x