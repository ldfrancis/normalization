"""Architecture tests for the ResNets in nstudy.core.model.

The reference is He et al., "Deep Residual Learning for Image Recognition"
(CVPR 2016). Section and table numbers below refer to that paper.
"""

from pathlib import Path

import pytest
import torch
from torch import nn

from nstudy.core.model import (
    BasicBlock,
    Bottleneck,
    ZeroPadShortcut,
    cifar_resnet,
    imagenet_resnet,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def modules_of(model: nn.Module, kind: type) -> list:
    return [m for m in model.modules() if isinstance(m, kind)]


def output_shapes(model: nn.Module, names: list[str], image_size: int) -> dict:
    """Run one image through the model and record (C, H, W) at each named module."""
    shapes, handles = {}, []
    for name in names:
        module = model.get_submodule(name)
        handles.append(
            module.register_forward_hook(
                lambda _m, _i, out, name=name: shapes.__setitem__(name, tuple(out.shape[1:]))
            )
        )
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, image_size, image_size))
    for handle in handles:
        handle.remove()
    return shapes


def count_flops(model: nn.Module, image_size: int) -> int:
    """Multiply-adds of the conv and fc layers, the quantity reported in Table 1."""
    total = 0

    def conv_hook(m, _inputs, out):
        nonlocal total
        kernel_ops = m.kernel_size[0] * m.kernel_size[1] * m.in_channels // m.groups
        total += kernel_ops * out.shape[1] * out.shape[2] * out.shape[3]

    def fc_hook(m, _inputs, _out):
        nonlocal total
        total += m.in_features * m.out_features

    handles = [m.register_forward_hook(conv_hook) for m in modules_of(model, nn.Conv2d)]
    handles += [m.register_forward_hook(fc_hook) for m in modules_of(model, nn.Linear)]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, image_size, image_size))
    for handle in handles:
        handle.remove()
    return total


# ---------------------------------------------------------------------------
# CIFAR ResNets (Section 4.2)
# ---------------------------------------------------------------------------

CIFAR_DEPTHS = [20, 32, 44, 56, 110]

# Exact counts follow from the Section 4.2 description with option A shortcuts
# and batch norm after every conv. The paper's Table 6 values are in comments.
CIFAR_PARAMS = {
    20: 269_722,  # 0.27M
    32: 464_154,  # 0.46M
    44: 658_586,  # 0.66M
    56: 853_018,  # 0.85M
    110: 1_727_962,  # 1.7M
}


@pytest.mark.parametrize("depth", CIFAR_DEPTHS)
def test_cifar_param_count(depth):
    assert count_params(cifar_resnet(depth)) == CIFAR_PARAMS[depth]


@pytest.mark.parametrize("depth", CIFAR_DEPTHS)
def test_cifar_has_6n_plus_2_weighted_layers(depth):
    """6n 3x3 convs in the stages, plus the first conv and the fc layer."""
    model = cifar_resnet(depth)
    n = (depth - 2) // 6
    assert len(modules_of(model, nn.Conv2d)) == 6 * n + 1
    assert len(modules_of(model, nn.Linear)) == 1
    for name in ("layer1", "layer2", "layer3"):
        stage = model.get_submodule(name)
        assert len(stage) == n
        assert all(isinstance(block, BasicBlock) for block in stage)
    assert not hasattr(model, "layer4")


@pytest.mark.parametrize("depth", [2, 18, 21, 34])
def test_cifar_rejects_depths_that_are_not_6n_plus_2(depth):
    with pytest.raises(ValueError):
        cifar_resnet(depth)


def test_cifar_stem_is_a_single_3x3_conv_with_16_filters():
    model = cifar_resnet(20)
    stem = model.conv1
    assert (stem.in_channels, stem.out_channels) == (3, 16)
    assert stem.kernel_size == (3, 3)
    assert stem.stride == (1, 1)
    assert stem.padding == (1, 1)
    assert not modules_of(model, nn.MaxPool2d)


def test_cifar_feature_map_sizes_and_widths():
    """Feature maps of size {32, 16, 8} with {16, 32, 64} filters."""
    shapes = output_shapes(
        cifar_resnet(20), ["conv1", "layer1", "layer2", "layer3", "avgpool"], image_size=32
    )
    assert shapes == {
        "conv1": (16, 32, 32),
        "layer1": (16, 32, 32),
        "layer2": (32, 16, 16),
        "layer3": (64, 8, 8),
        "avgpool": (64, 1, 1),
    }


def test_cifar_uses_only_3x3_convs_and_subsamples_twice():
    """Every conv is 3x3; subsampling is a stride-2 conv opening stages 2 and 3."""
    model = cifar_resnet(20)
    convs = modules_of(model, nn.Conv2d)
    assert all(conv.kernel_size == (3, 3) for conv in convs)
    assert all(conv.bias is None for conv in convs)
    strided = [name for name, m in model.named_modules()
               if isinstance(m, nn.Conv2d) and m.stride == (2, 2)]
    assert strided == ["layer2.0.conv1", "layer3.0.conv1"]


def test_cifar_shortcuts_are_parameter_free():
    """'We use identity shortcuts in all cases (i.e., option A)'."""
    model = cifar_resnet(20)
    for name in ("layer1", "layer2", "layer3"):
        for index, block in enumerate(model.get_submodule(name)):
            if name != "layer1" and index == 0:
                assert isinstance(block.downsample, ZeroPadShortcut)
                assert count_params(block.downsample) == 0
            else:
                assert block.downsample is None


def test_zero_pad_shortcut_subsamples_and_pads_channels_with_zeros():
    x = torch.randn(2, 16, 32, 32)
    out = ZeroPadShortcut(16, 32, stride=2)(x)
    assert out.shape == (2, 32, 16, 16)
    assert torch.equal(out[:, :16], x[:, :, ::2, ::2])
    assert torch.count_nonzero(out[:, 16:]) == 0


@pytest.mark.parametrize("num_classes", [10, 100])
def test_cifar_classifier_head(num_classes):
    model = cifar_resnet(20, num_classes=num_classes)
    assert (model.fc.in_features, model.fc.out_features) == (64, num_classes)
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, num_classes)


# ---------------------------------------------------------------------------
# ImageNet ResNets (Section 3.3 and Table 1)
# ---------------------------------------------------------------------------

IMAGENET_DEPTHS = [18, 34, 50, 101, 152]

# Table 1: block type and number of blocks in conv2_x .. conv5_x.
IMAGENET_BLOCKS = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3]),
}

# Table 1, bottom row, in units of 1e9 multiply-adds.
IMAGENET_FLOPS = {18: 1.8, 34: 3.6, 50: 3.8, 101: 7.6, 152: 11.3}

# The paper does not list ImageNet parameter counts. These are the counts of
# the standard option-B models (identical to torchvision's).
IMAGENET_PARAMS = {
    18: 11_689_512,
    34: 21_797_672,
    50: 25_557_032,
    101: 44_549_160,
    152: 60_192_808,
}

STAGES = ["layer1", "layer2", "layer3", "layer4"]


@pytest.mark.parametrize("depth", IMAGENET_DEPTHS)
def test_imagenet_block_type_and_stage_depths(depth):
    model = imagenet_resnet(depth)
    block_type, blocks = IMAGENET_BLOCKS[depth]
    assert [len(model.get_submodule(name)) for name in STAGES] == blocks
    for name in STAGES:
        assert all(type(block) is block_type for block in model.get_submodule(name))


@pytest.mark.parametrize("depth", IMAGENET_DEPTHS)
def test_imagenet_weighted_layer_count_equals_depth(depth):
    """The depth counts convs and the fc layer, not the projection shortcuts."""
    model = imagenet_resnet(depth)
    convs = [name for name, m in model.named_modules()
             if isinstance(m, nn.Conv2d) and "downsample" not in name]
    assert len(convs) + len(modules_of(model, nn.Linear)) == depth


@pytest.mark.parametrize("depth", IMAGENET_DEPTHS)
def test_imagenet_param_count(depth):
    assert count_params(imagenet_resnet(depth)) == IMAGENET_PARAMS[depth]


@pytest.mark.parametrize("depth", IMAGENET_DEPTHS)
def test_imagenet_flops_match_table_1(depth):
    flops = count_flops(imagenet_resnet(depth), image_size=224) / 1e9
    assert flops == pytest.approx(IMAGENET_FLOPS[depth], abs=0.1)


@pytest.mark.parametrize("depth", [20, 56, 100])
def test_imagenet_rejects_unknown_depths(depth):
    with pytest.raises(ValueError):
        imagenet_resnet(depth)


def test_imagenet_stem_is_7x7_stride_2_conv_then_3x3_stride_2_max_pool():
    model = imagenet_resnet(18)
    stem = model.conv1
    assert (stem.in_channels, stem.out_channels) == (3, 64)
    assert stem.kernel_size == (7, 7)
    assert stem.stride == (2, 2)
    assert stem.padding == (3, 3)
    assert isinstance(model.maxpool, nn.MaxPool2d)
    assert (model.maxpool.kernel_size, model.maxpool.stride) == (3, 2)


@pytest.mark.parametrize("depth, expansion", [(18, 1), (50, 4)])
def test_imagenet_feature_map_sizes_and_widths(depth, expansion):
    """Output sizes 112, 56, 28, 14, 7 from the 'output size' column of Table 1."""
    shapes = output_shapes(
        imagenet_resnet(depth), ["conv1", "maxpool", *STAGES, "avgpool"], image_size=224
    )
    assert shapes == {
        "conv1": (64, 112, 112),
        "maxpool": (64, 56, 56),
        "layer1": (64 * expansion, 56, 56),
        "layer2": (128 * expansion, 28, 28),
        "layer3": (256 * expansion, 14, 14),
        "layer4": (512 * expansion, 7, 7),
        "avgpool": (512 * expansion, 1, 1),
    }


def test_basic_block_is_two_3x3_convs():
    block = imagenet_resnet(18).layer2[1]
    assert [c.kernel_size for c in (block.conv1, block.conv2)] == [(3, 3), (3, 3)]
    assert [c.out_channels for c in (block.conv1, block.conv2)] == [128, 128]


def test_bottleneck_block_is_1x1_3x3_1x1_with_4x_expansion():
    block = imagenet_resnet(50).layer2[1]
    convs = (block.conv1, block.conv2, block.conv3)
    assert [c.kernel_size for c in convs] == [(1, 1), (3, 3), (1, 1)]
    assert [c.in_channels for c in convs] == [512, 128, 128]
    assert [c.out_channels for c in convs] == [128, 128, 512]


@pytest.mark.parametrize("depth", [18, 50])
def test_imagenet_downsamples_in_first_block_of_conv3_conv4_conv5(depth):
    """Stride 2 sits on the first conv of the block, as in the paper."""
    model = imagenet_resnet(depth)
    strided = [name for name, m in model.named_modules()
               if isinstance(m, nn.Conv2d) and m.stride == (2, 2) and "downsample" not in name]
    assert strided == ["conv1", "layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"]


def test_bottleneck_stride_can_move_to_the_3x3_conv():
    """The torchvision ('v1.5') variant, needed to load torchvision weights."""
    model = imagenet_resnet(50, stride_in_3x3=True)
    block = model.layer2[0]
    assert block.conv1.stride == (1, 1)
    assert block.conv2.stride == (2, 2)
    assert count_params(model) == IMAGENET_PARAMS[50]


@pytest.mark.parametrize("depth", [18, 50])
def test_imagenet_uses_projection_shortcuts_only_where_dimensions_change(depth):
    """Option B: 1x1 conv + norm when the shape changes, identity otherwise."""
    model = imagenet_resnet(depth)
    for name in STAGES:
        for index, block in enumerate(model.get_submodule(name)):
            needs_projection = index == 0 and not (depth == 18 and name == "layer1")
            if needs_projection:
                conv = block.downsample[0]
                assert isinstance(conv, nn.Conv2d)
                assert conv.kernel_size == (1, 1)
                assert isinstance(block.downsample[1], nn.BatchNorm2d)
            else:
                assert block.downsample is None


def test_imagenet_classifier_head():
    model = imagenet_resnet(50)
    assert (model.fc.in_features, model.fc.out_features) == (2048, 1000)
    assert isinstance(model.avgpool, nn.AdaptiveAvgPool2d)


@pytest.mark.parametrize("depth", IMAGENET_DEPTHS)
def test_imagenet_state_dict_layout_matches_torchvision(depth):
    """Same keys and shapes, so torchvision checkpoints load directly."""
    torchvision = pytest.importorskip("torchvision")
    ours = imagenet_resnet(depth).state_dict()
    theirs = getattr(torchvision.models, f"resnet{depth}")(weights=None).state_dict()
    assert {k: v.shape for k, v in ours.items()} == {k: v.shape for k, v in theirs.items()}


# ---------------------------------------------------------------------------
# Shared behaviour
# ---------------------------------------------------------------------------

BUILDERS = [(cifar_resnet, 20), (imagenet_resnet, 18), (imagenet_resnet, 50)]


@pytest.mark.parametrize("builder, depth", BUILDERS)
def test_every_conv_is_followed_by_batch_norm_by_default(builder, depth):
    model = builder(depth)
    convs = modules_of(model, nn.Conv2d)
    norms = modules_of(model, nn.BatchNorm2d)
    assert len(norms) == len(convs)


@pytest.mark.parametrize("builder, depth", BUILDERS)
def test_norm_layer_is_pluggable(builder, depth):
    model = builder(depth, norm_layer=lambda channels: nn.GroupNorm(1, channels))
    assert not modules_of(model, nn.BatchNorm2d)
    assert len(modules_of(model, nn.GroupNorm)) == len(modules_of(model, nn.Conv2d))


# ---------------------------------------------------------------------------
# Hydra configs
# ---------------------------------------------------------------------------


def build_from_config(overrides: list[str]) -> nn.Module:
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return hydra.utils.instantiate(cfg.model)


@pytest.mark.parametrize(
    "overrides, expected_params",
    [
        (["model=cifar_resnet", "model.depth=20"], CIFAR_PARAMS[20]),
        (["model=cifar_resnet", "model.depth=56"], CIFAR_PARAMS[56]),
        (["model=imagenet_resnet", "model.depth=18"], IMAGENET_PARAMS[18]),
        (["model=imagenet_resnet", "model.depth=50"], IMAGENET_PARAMS[50]),
    ],
)
def test_hydra_config_builds_the_requested_model(overrides, expected_params):
    assert count_params(build_from_config(overrides)) == expected_params


def test_hydra_config_overrides_num_classes_for_cifar100():
    model = build_from_config(["model=cifar_resnet", "model.num_classes=100"])
    assert model.fc.out_features == 100


# ---------------------------------------------------------------------------
# Plain nets: the same layers without shortcuts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build, depth", [(cifar_resnet, 20), (cifar_resnet, 56),
                                          (imagenet_resnet, 18), (imagenet_resnet, 50)])
def test_plain_net_has_the_layers_of_the_resnet_and_no_shortcuts(build, depth):
    resnet, plain = build(depth), build(depth, residual=False)
    blocks = [m for m in plain.modules() if isinstance(m, (BasicBlock, Bottleneck))]
    assert blocks and all(not b.residual and b.downsample is None for b in blocks)
    # Every layer of the ResNet but its projection shortcuts (option B) is there, same shape.
    shapes = {name: p.shape for name, p in resnet.named_parameters() if "downsample" not in name}
    assert {name: p.shape for name, p in plain.named_parameters()} == shapes
    out = plain(torch.randn(2, 3, 32, 32) if build is cifar_resnet else torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 10 if build is cifar_resnet else 1000)


@pytest.mark.parametrize("build, depth", [(cifar_resnet, 56), (imagenet_resnet, 34)])
def test_plain_net_convs_use_kaiming_init(build, depth):
    """Plain nets get the same He et al. (2015) init as the ResNets: std sqrt(2 / fan_out)."""
    torch.manual_seed(0)
    convs = [m for m in build(depth, residual=False).modules() if isinstance(m, nn.Conv2d)]
    assert convs
    for conv in convs:
        fan_out = conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1]
        assert conv.weight.std().item() == pytest.approx((2 / fan_out) ** 0.5, rel=0.1)


@pytest.mark.parametrize("block, in_channels, stride", [
    (BasicBlock, 16, 1), (BasicBlock, 16, 2), (Bottleneck, 64, 1), (Bottleneck, 64, 2)])
def test_plain_block_adds_nothing_back(block, in_channels, stride):
    """A plain block is its residual block's output before the addition, then a ReLU."""
    torch.manual_seed(0)
    plain = block(in_channels, 16, stride, nn.BatchNorm2d, "B", residual=False).eval()
    x = torch.randn(2, in_channels, 8, 8)
    out = plain.relu(plain.bn1(plain.conv1(x)))
    if block is Bottleneck:
        out = plain.relu(plain.bn2(plain.conv2(out)))
        out = plain.bn3(plain.conv3(out))
    else:
        out = plain.bn2(plain.conv2(out))
    assert torch.equal(plain(x), torch.relu(out))


def test_plain_nets_are_chosen_in_the_config():
    hydra = pytest.importorskip("hydra")
    from hydra.utils import instantiate

    for experiment in ("cifar10_resnet", "imagenet_resnet"):
        with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = hydra.compose(config_name="config",
                                overrides=[f"+experiment={experiment}", "model.residual=false"])
        model = instantiate(cfg.model)
        assert not model.residual and not any(
            getattr(m, "residual", False) for m in model.modules())
