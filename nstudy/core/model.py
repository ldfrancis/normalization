"""Models for the normalization study.

ResNets from He et al., "Deep Residual Learning for Image Recognition" (2016):

* ``cifar_resnet(depth)``: the 6n+2 networks of Section 4.2 (ResNet-20/32/44/56/110).
* ``imagenet_resnet(depth)``: the networks of Table 1 (ResNet-18/34/50/101/152).

The Transformer from Vaswani et al., "Attention Is All You Need" (2017):

* ``Transformer(vocab_size)``: the base model by default; Table 3 gives the big one.

Every normalization layer is created through ``norm_layer``, a callable that
takes the number of channels (or the model dimension) and returns a module.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

NormLayer = Callable[[int], nn.Module]


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)


class ZeroPadShortcut(nn.Module):
    """Option A shortcut: subsample spatially, pad the new channels with zeros."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.extra_channels = out_channels - in_channels

    def forward(self, x: Tensor) -> Tensor:
        x = x[:, :, :: self.stride, :: self.stride]
        return F.pad(x, (0, 0, 0, 0, 0, self.extra_channels))


def make_shortcut(
    kind: str, in_channels: int, out_channels: int, stride: int, norm_layer: NormLayer
) -> nn.Module | None:
    """Return the shortcut module, or None when the identity can be used as is."""
    if stride == 1 and in_channels == out_channels:
        return None
    if kind == "A":
        return ZeroPadShortcut(in_channels, out_channels, stride)
    if kind == "B":
        return nn.Sequential(conv1x1(in_channels, out_channels, stride), norm_layer(out_channels))
    raise ValueError(f"shortcut must be 'A' or 'B', got {kind!r}")


class BasicBlock(nn.Module):
    """Two 3x3 convs (Figure 5, left)."""

    expansion = 1

    def __init__(
        self, in_channels: int, width: int, stride: int, norm_layer: NormLayer, shortcut: str
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, width, stride)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = make_shortcut(shortcut, in_channels, width, stride, norm_layer)

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    """1x1, 3x3, 1x1 convs with a 4x wider output (Figure 5, right).

    The paper puts the stride on the first 1x1 conv. ``stride_in_3x3=True``
    moves it to the 3x3 conv, which is what torchvision does.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        width: int,
        stride: int,
        norm_layer: NormLayer,
        shortcut: str,
        stride_in_3x3: bool = False,
    ) -> None:
        super().__init__()
        out_channels = width * self.expansion
        stride_1x1, stride_3x3 = (1, stride) if stride_in_3x3 else (stride, 1)
        self.conv1 = conv1x1(in_channels, width, stride_1x1)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride_3x3)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels)
        self.bn3 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = make_shortcut(shortcut, in_channels, out_channels, stride, norm_layer)

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNet(nn.Module):
    """Stem, a sequence of residual stages, global average pooling and a linear head.

    Module names follow torchvision (``conv1``, ``bn1``, ``layer1``, ..., ``fc``)
    so that torchvision checkpoints load into the ImageNet models.

    Args:
        block: ``BasicBlock`` or ``Bottleneck``.
        blocks_per_stage: number of blocks in each stage.
        widths: block width of each stage; the first is also the stem width.
        num_classes: size of the classifier output.
        stem: ``"cifar"`` (3x3 conv) or ``"imagenet"`` (7x7 stride-2 conv + max pool).
        shortcut: ``"A"`` (zero padding) or ``"B"`` (1x1 projection) where shapes change.
        norm_layer: callable mapping a channel count to a normalization module.
        block_kwargs: extra keyword arguments passed to every block.
    """

    def __init__(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        blocks_per_stage: list[int],
        widths: list[int],
        num_classes: int,
        stem: str,
        shortcut: str,
        norm_layer: NormLayer = nn.BatchNorm2d,
        block_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if len(blocks_per_stage) != len(widths):
            raise ValueError("blocks_per_stage and widths must have the same length")
        block_kwargs = block_kwargs or {}

        if stem == "cifar":
            self.conv1 = conv3x3(3, widths[0])
            self.maxpool = nn.Identity()
        elif stem == "imagenet":
            self.conv1 = nn.Conv2d(3, widths[0], 7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f"stem must be 'cifar' or 'imagenet', got {stem!r}")
        self.bn1 = norm_layer(widths[0])
        self.relu = nn.ReLU(inplace=True)

        self.stage_names = []
        in_channels = widths[0]
        for index, (num_blocks, width) in enumerate(zip(blocks_per_stage, widths)):
            blocks = []
            for block_index in range(num_blocks):
                # Every stage after the first halves the resolution in its first block.
                stride = 2 if index > 0 and block_index == 0 else 1
                blocks.append(
                    block(in_channels, width, stride, norm_layer, shortcut, **block_kwargs)
                )
                in_channels = width * block.expansion
            name = f"layer{index + 1}"
            setattr(self, name, nn.Sequential(*blocks))
            self.stage_names.append(name)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        # Norm layers that start from the fan-in of their input (such as
        # nstudy.core.norms.DynamicScale) learn it from the conv before them.
        for conv, norm in self.conv_norm_pairs():
            set_fan_in = getattr(norm, "set_fan_in", None)
            if callable(set_fan_in):
                kernel = conv.kernel_size[0] * conv.kernel_size[1]
                set_fan_in(conv.in_channels // conv.groups * kernel)

    def conv_norm_pairs(self) -> list[tuple[nn.Conv2d, nn.Module]]:
        """Every norm layer with the conv whose output it normalizes."""
        pairs = [(self.conv1, self.bn1)]
        for name in self.stage_names:
            for block in getattr(self, name):
                for index in (1, 2, 3):
                    conv = getattr(block, f"conv{index}", None)
                    if isinstance(conv, nn.Conv2d):
                        pairs.append((conv, getattr(block, f"bn{index}")))
                downsample = block.downsample
                if isinstance(downsample, nn.Sequential):  # option B: 1x1 conv, norm
                    conv = downsample[0]
                    if isinstance(conv, nn.Conv2d):
                        pairs.append((conv, downsample[1]))
        return pairs

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for name in self.stage_names:
            x = getattr(self, name)(x)
        return self.fc(self.avgpool(x).flatten(1))


IMAGENET_ARCHITECTURES: dict[int, tuple[type, list[int]]] = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3]),
}


def cifar_resnet(
    depth: int,
    num_classes: int = 10,
    norm_layer: NormLayer = nn.BatchNorm2d,
    shortcut: str = "A",
) -> ResNet:
    """ResNet-``depth`` for 32x32 inputs, with ``depth = 6n + 2`` (Section 4.2)."""
    if depth < 8 or (depth - 2) % 6 != 0:
        raise ValueError(f"CIFAR ResNet depth must be 6n+2 with n >= 1, got {depth}")
    n = (depth - 2) // 6
    return ResNet(
        BasicBlock,
        blocks_per_stage=[n, n, n],
        widths=[16, 32, 64],
        num_classes=num_classes,
        stem="cifar",
        shortcut=shortcut,
        norm_layer=norm_layer,
    )


def imagenet_resnet(
    depth: int,
    num_classes: int = 1000,
    norm_layer: NormLayer = nn.BatchNorm2d,
    shortcut: str = "B",
    stride_in_3x3: bool = False,
) -> ResNet:
    """ResNet-``depth`` for 224x224 inputs, with depth in {18, 34, 50, 101, 152} (Table 1)."""
    if depth not in IMAGENET_ARCHITECTURES:
        raise ValueError(
            f"ImageNet ResNet depth must be one of {sorted(IMAGENET_ARCHITECTURES)}, got {depth}"
        )
    block, blocks_per_stage = IMAGENET_ARCHITECTURES[depth]
    return ResNet(
        block,
        blocks_per_stage=blocks_per_stage,
        widths=[64, 128, 256, 512],
        num_classes=num_classes,
        stem="imagenet",
        shortcut=shortcut,
        norm_layer=norm_layer,
        block_kwargs={"stride_in_3x3": stride_in_3x3} if block is Bottleneck else None,
    )


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


def sinusoidal_positions(length: int, d_model: int, device: torch.device | None = None) -> Tensor:
    """Positional encodings of Section 3.5, shape ``(length, d_model)``.

    ``PE[pos, 2i] = sin(pos / 10000^(2i / d_model))`` and
    ``PE[pos, 2i + 1] = cos(pos / 10000^(2i / d_model))``.
    """
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    even_dims = torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
    angles = position * torch.exp(even_dims * (-math.log(10000.0) / d_model))
    table = torch.zeros(length, d_model, device=device)
    table[:, 0::2] = torch.sin(angles)
    table[:, 1::2] = torch.cos(angles)
    return table


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention over ``num_heads`` heads (Section 3.2)."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: Tensor, memory: Tensor, mask: Tensor | None) -> Tensor:
        """Attend from ``query`` (B, T, D) to ``memory`` (B, S, D).

        ``mask`` is boolean, broadcastable to (B, heads, T, S), and True where
        attention is allowed.
        """
        batch, tgt_len, d_model = query.shape

        def split_heads(x: Tensor) -> Tensor:
            return x.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(query))
        k = split_heads(self.k_proj(memory))
        v = split_heads(self.v_proj(memory))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(out.transpose(1, 2).reshape(batch, tgt_len, d_model))


class EncoderLayer(nn.Module):
    """Self-attention then feed-forward.

    Post-norm (the paper) wraps each sub-layer as ``norm(x + dropout(sublayer(x)))``.
    Pre-norm (``norm_first=True``) wraps it as ``x + dropout(sublayer(norm(x)))``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        norm_layer: NormLayer,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = norm_layer(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm2 = norm_layer(d_model)
        self.dropout = nn.Dropout(dropout)

    def feed_forward(self, x: Tensor) -> Tensor:
        return self.linear2(F.relu(self.linear1(x)))

    def forward(self, x: Tensor, src_mask: Tensor | None) -> Tensor:
        if self.norm_first:
            y = self.norm1(x)
            x = x + self.dropout(self.self_attn(y, y, src_mask))
            return x + self.dropout(self.feed_forward(self.norm2(x)))
        x = self.norm1(x + self.dropout(self.self_attn(x, x, src_mask)))
        return self.norm2(x + self.dropout(self.feed_forward(x)))


class DecoderLayer(nn.Module):
    """Masked self-attention, attention over the encoder output, then feed-forward.

    ``norm_first`` has the same meaning as in ``EncoderLayer``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        norm_layer: NormLayer,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = norm_layer(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = norm_layer(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm3 = norm_layer(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, memory: Tensor, tgt_mask: Tensor | None, src_mask: Tensor | None
    ) -> Tensor:
        if self.norm_first:
            y = self.norm1(x)
            x = x + self.dropout(self.self_attn(y, y, tgt_mask))
            x = x + self.dropout(self.cross_attn(self.norm2(x), memory, src_mask))
            return x + self.dropout(self.feed_forward(self.norm3(x)))
        x = self.norm1(x + self.dropout(self.self_attn(x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, src_mask)))
        return self.norm3(x + self.dropout(self.feed_forward(x)))

    def feed_forward(self, x: Tensor) -> Tensor:
        return self.linear2(F.relu(self.linear1(x)))


class Transformer(nn.Module):
    """Encoder-decoder Transformer for translation with a shared vocabulary.

    The source embedding, the target embedding and the pre-softmax projection
    share one weight matrix (Section 3.4). The defaults give the base model.

    Args:
        vocab_size: size of the shared source-target vocabulary.
        d_model: model dimension.
        num_heads: attention heads per attention sub-layer.
        num_layers: number of encoder layers and of decoder layers.
        d_ff: inner dimension of the feed-forward sub-layers.
        dropout: applied to each sub-layer output and to the embedded inputs.
        pad_id: token id of padding; padded source positions are not attended to.
        norm_layer: callable mapping ``d_model`` to a normalization module.
        norm_first: False normalizes each sub-layer's output after the residual
            sum, as in the paper (post-norm). True normalizes each sub-layer's
            input instead (pre-norm) and adds a final norm after the encoder
            stack and after the decoder stack.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_id: int = 0,
        norm_layer: NormLayer = nn.LayerNorm,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sinusoidal positions, got {d_model}")
        self.d_model = d_model
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            EncoderLayer(d_model, num_heads, d_ff, dropout, norm_layer, norm_first)
            for _ in range(num_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderLayer(d_model, num_heads, d_ff, dropout, norm_layer, norm_first)
            for _ in range(num_layers)
        )
        # With pre-norm the residual stream leaving each stack is unnormalized.
        self.encoder_norm = norm_layer(d_model) if norm_first else None
        self.decoder_norm = norm_layer(d_model) if norm_first else None
        self.generator = nn.Linear(d_model, vocab_size, bias=False)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Tie after the loop so the shared matrix keeps the embedding initialisation.
        nn.init.normal_(self.embedding.weight, std=d_model**-0.5)
        with torch.no_grad():
            self.embedding.weight[pad_id].zero_()
        self.generator.weight = self.embedding.weight

    def embed(self, tokens: Tensor) -> Tensor:
        """Scaled token embeddings plus positional encodings, shape (B, L, D)."""
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = x + sinusoidal_positions(tokens.shape[1], self.d_model, x.device)
        return self.dropout(x)

    def source_mask(self, src: Tensor) -> Tensor:
        """True at real source tokens, shape (B, 1, 1, S)."""
        return (src != self.pad_id)[:, None, None, :]

    def encode(self, src: Tensor) -> Tensor:
        """Encode source token ids (B, S) into (B, S, D)."""
        src_mask = self.source_mask(src)
        x = self.embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x if self.encoder_norm is None else self.encoder_norm(x)

    def decode(self, tgt: Tensor, memory: Tensor, src: Tensor) -> Tensor:
        """Return logits (B, T, V) for target token ids (B, T) given the encoded source.

        Position ``t`` only sees target positions up to ``t``. Target padding
        needs no mask as long as it comes after the real tokens.
        """
        tgt_len = tgt.shape[1]
        tgt_mask = torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device).tril()
        src_mask = self.source_mask(src)
        x = self.embed(tgt)
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, src_mask)
        if self.decoder_norm is not None:
            x = self.decoder_norm(x)
        return self.generator(x)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        return self.decode(tgt, self.encode(src), src)
