"""Tests for the element-wise replacements of normalization layers."""

from functools import partial
from pathlib import Path

import pytest
import torch
from torch import nn

from nstudy.core import evaluator, norm_io
from nstudy.core.model import Transformer, cifar_resnet, imagenet_resnet
from nstudy.core.norms import DynamicScale, DynamicSoftsign, DynamicTanh, ElementwiseNorm
from nstudy.core.probes import model_probes
from nstudy.core.tracking import run_tags

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
CPU = torch.device("cpu")


def randomized(layer: ElementwiseNorm) -> ElementwiseNorm:
    with torch.no_grad():
        layer.weight.uniform_(0.5, 2)
        layer.bias.uniform_(-1, 1)
    return layer


def test_dynamic_tanh_on_images_scales_and_shifts_along_channels():
    layer = randomized(DynamicTanh(4, alpha_init=0.7))
    x = torch.randn(2, 4, 3, 3) * 3
    gamma, beta = layer.weight[None, :, None, None], layer.bias[None, :, None, None]
    assert torch.allclose(layer(x), gamma * torch.tanh(0.7 * x) + beta)


def test_dynamic_tanh_on_tokens_scales_and_shifts_along_features():
    layer = randomized(DynamicTanh(4))
    x = torch.randn(2, 5, 4) * 3
    assert torch.allclose(layer(x), layer.weight * torch.tanh(0.5 * x) + layer.bias)


def test_dynamic_softsign_matches_its_formula():
    layer = randomized(DynamicSoftsign(4))
    x = torch.randn(2, 5, 4) * 3
    assert torch.allclose(layer(x), layer.weight * x / (2.0 + x.abs()) + layer.bias)
    with torch.no_grad():
        layer.alpha.fill_(-2.0)  # |alpha|: the denominator stays positive
    assert torch.allclose(layer(x), layer.weight * x / (2.0 + x.abs()) + layer.bias)


def test_softsign_default_starts_with_the_slope_of_dyt():
    x = torch.tensor([0.0], requires_grad=True)
    slopes = []
    for layer in (DynamicTanh(1), DynamicSoftsign(1)):
        (slope,) = torch.autograd.grad(layer(x.view(1, 1)).sum(), x)
        slopes.append(slope.item())
    assert slopes == pytest.approx([0.5, 0.5])


def test_alpha_per_layer_or_per_channel_and_every_parameter_learns():
    assert DynamicTanh(8).alpha.shape == (1,)
    layer = DynamicTanh(8, per_channel_alpha=True)
    assert layer.alpha.shape == (8,)
    with torch.no_grad():
        layer.alpha.copy_(torch.linspace(0.1, 1.0, 8))
    x = torch.randn(3, 8, 2, 2)
    assert torch.allclose(layer(x), torch.tanh(layer.alpha[None, :, None, None] * x))
    layer(x).pow(2).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters())


def test_dynamic_scale_multiplies_by_alpha_starting_at_one_over_sqrt_d_model():
    layer = randomized(DynamicScale(16))
    assert layer.alpha.item() == pytest.approx(0.25)  # 1 / sqrt(16)
    x = torch.randn(2, 5, 16)
    assert torch.allclose(layer(x), layer.weight * 0.25 * x + layer.bias)
    assert DynamicScale(16, alpha_init=0.3).alpha.item() == pytest.approx(0.3)


def test_alpha_of_dynamic_scale_gets_a_usable_gradient():
    """d(alpha * x)/d(alpha) = x, unlike d(x / alpha)/d(alpha) = -x / alpha**2."""
    layer = DynamicScale(1)
    x = torch.full((1, 1), 3.0)
    (grad,) = torch.autograd.grad(layer(x).sum(), layer.alpha)
    assert grad.item() == pytest.approx(3.0)


def test_transformer_scale_layers_start_at_one_over_sqrt_d_model():
    model = Transformer(vocab_size=30, d_model=64, num_heads=2, num_layers=2, d_ff=32,
                        norm_layer=DynamicScale)
    alphas = [m.alpha.item() for m in model.modules() if isinstance(m, DynamicScale)]
    assert len(alphas) == 2 * 2 + 2 * 3 and alphas == pytest.approx([1 / 8] * len(alphas))


def test_resnet_scale_layers_start_at_one_over_sqrt_of_the_fan_in_of_their_conv():
    model = cifar_resnet(depth=8, norm_layer=DynamicScale)
    alpha = {name: m.alpha.item() for name, m in model.named_modules()
             if isinstance(m, DynamicScale)}
    assert alpha["bn1"] == pytest.approx(27 ** -0.5)  # stem: 3 channels x 3 x 3
    assert alpha["layer1.0.bn1"] == pytest.approx(1 / 12)  # 16 x 3 x 3
    assert alpha["layer2.0.bn1"] == pytest.approx(1 / 12)  # the conv takes 16 channels in
    assert alpha["layer2.0.bn2"] == pytest.approx(288 ** -0.5)  # 32 x 3 x 3

    model = imagenet_resnet(depth=50, norm_layer=DynamicScale)
    alpha = {name: m.alpha.item() for name, m in model.named_modules()
             if isinstance(m, DynamicScale)}
    assert alpha["bn1"] == pytest.approx(147 ** -0.5)  # stem: 3 x 7 x 7
    assert alpha["layer1.0.bn1"] == pytest.approx(1 / 8)  # 1x1 conv, 64 in
    assert alpha["layer1.0.bn2"] == pytest.approx(1 / 24)  # 3x3 conv, 64 in
    assert alpha["layer1.0.downsample.1"] == pytest.approx(1 / 8)  # 1x1 projection, 64 in
    norms = {id(m) for m in model.modules() if isinstance(m, DynamicScale)}
    assert {id(norm) for _, norm in model.conv_norm_pairs()} == norms  # every norm has its conv


def test_resnet_keeps_a_given_alpha_init():
    model = cifar_resnet(depth=8, norm_layer=partial(DynamicScale, alpha_init=1.0))
    assert all(m.alpha.item() == 1.0 for m in model.modules() if isinstance(m, DynamicScale))


def test_alpha_and_gamma_get_weight_decay_like_every_parameter():
    """With zero gradients, an SGD step only applies weight decay: everything shrinks."""
    model = cifar_resnet(depth=8, norm_layer=DynamicScale)
    layers = [m for m in model.modules() if isinstance(m, DynamicScale)]
    before = [(m.alpha.clone(), m.weight.clone()) for m in layers]
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, weight_decay=0.1)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    for m, (alpha, gamma) in zip(layers, before):
        assert torch.allclose(m.alpha, 0.9 * alpha) and torch.allclose(m.weight, 0.9 * gamma)


@pytest.mark.parametrize("norm", [DynamicTanh, DynamicSoftsign, DynamicScale])
def test_beta_can_be_left_out(norm):
    layer = norm(4, bias=False)
    assert layer.bias is None
    assert [name for name, _ in layer.named_parameters()] == ["alpha", "weight"]
    x = torch.randn(2, 4, 3, 3)
    with torch.no_grad():
        layer.weight.uniform_(0.5, 2)
    expected = layer.squash(x, layer.per_channel(layer.alpha, x)) * layer.weight[None, :, None, None]
    assert torch.allclose(layer(x), expected)
    model = cifar_resnet(depth=8, norm_layer=partial(norm, bias=False))
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    # Only the classifier is left with a bias: the convs have none.
    assert [name for name, _ in model.named_parameters() if name.endswith("bias")] == ["fc.bias"]


def test_bias_option_reaches_the_layer_from_the_config():
    hydra = pytest.importorskip("hydra")
    from hydra.utils import instantiate

    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=[
            "+experiment=cifar10_resnet", "norm=dyscale", "model.norm_layer.bias=false"])
    assert instantiate(cfg.model.norm_layer)(16).bias is None


@pytest.mark.parametrize("norm", [DynamicTanh, DynamicSoftsign, DynamicScale, nn.Identity])
def test_models_train_with_each_replacement(norm):
    torch.manual_seed(0)
    resnet = cifar_resnet(depth=8, norm_layer=norm)
    transformer = Transformer(vocab_size=30, d_model=16, num_heads=2, num_layers=2, d_ff=32,
                              norm_layer=norm)
    assert not any(isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)) for m in resnet.modules())
    assert not any(isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)) for m in transformer.modules())
    loss = nn.functional.cross_entropy(resnet(torch.randn(4, 3, 32, 32)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert torch.isfinite(loss)
    src = torch.tensor([[5, 6, 7], [8, 9, 0]])
    logits = transformer(src, src)
    assert logits.shape == (2, 3, 30) and torch.isfinite(logits).all()


@pytest.mark.parametrize("norm", ["dyt", "dysoftsign", "dyscale", "none", "batchnorm", "layernorm"])
@pytest.mark.parametrize("experiment", ["cifar10_resnet", "wmt14_transformer_base"])
def test_norm_option_replaces_the_norm_layer_of_any_experiment(norm, experiment):
    hydra = pytest.importorskip("hydra")
    from hydra.utils import instantiate

    if (norm, experiment) in {("batchnorm", "wmt14_transformer_base"),
                              ("layernorm", "cifar10_resnet")}:
        pytest.skip("BatchNorm2d needs images; LayerNorm here normalizes tokens")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=[f"+experiment={experiment}",
                                                             f"norm={norm}"])
    expected = {"dyt": DynamicTanh, "dysoftsign": DynamicSoftsign, "dyscale": DynamicScale,
                "none": nn.Identity,
                "batchnorm": nn.BatchNorm2d, "layernorm": nn.LayerNorm}[norm]
    layer = instantiate(cfg.model.norm_layer)(16)
    assert type(layer) is expected
    assert run_tags(cfg, {})[-1] == expected.__name__


def test_without_the_option_the_models_keep_their_own_norms():
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        resnet = hydra.compose(config_name="config", overrides=["+experiment=cifar10_resnet"])
        transformer = hydra.compose(config_name="config",
                                    overrides=["+experiment=wmt14_transformer_base"])
    assert resnet.model.norm_layer._target_ == "torch.nn.BatchNorm2d"
    assert transformer.model.norm_layer._target_ == "torch.nn.LayerNorm"


# ---------------------------------------------------------------------------
# The norm_io analysis on these layers
# ---------------------------------------------------------------------------


def recorded(model, batch, task):
    recorder = norm_io.NormIORecorder(model, model_probes(model), points_per_batch=10**9)
    recorder.run([batch], task, CPU, examples=1)
    recorder.remove()
    return {name: r.tensors() for name, r in recorder.records.items()}, recorder


def test_norm_io_shows_tanh_of_alpha_x_for_dyt():
    torch.manual_seed(0)
    model = cifar_resnet(depth=8, norm_layer=partial(DynamicTanh, per_channel_alpha=True))
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, DynamicTanh):
                randomized(module).alpha.uniform_(0.2, 2)
    batch = (torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
    layers, recorder = recorded(model, batch, evaluator.classification())
    for probe in recorder.probes:
        points, alpha = layers[probe.name], probe.module.alpha
        c = points["channel"]
        assert torch.allclose(points["x_hat"], torch.tanh(alpha[c] * points["x"]), atol=1e-6)
        assert torch.allclose(points["y"], probe.module.weight[c] * points["x_hat"]
                              + probe.module.bias[c], atol=1e-5)


def test_norm_io_on_a_transformer_with_softsign_and_on_no_norm():
    torch.manual_seed(0)
    src = torch.tensor([[5, 6, 7, 0]])
    batch = {"src": src, "tgt_in": src, "tgt_out": src}
    model = Transformer(vocab_size=30, d_model=16, num_heads=2, num_layers=3, d_ff=32,
                        norm_layer=DynamicSoftsign)
    layers, _ = recorded(model, batch, evaluator.translation())
    for points in layers.values():
        assert points["x"].numel() == 3 * 16  # padding left out
        assert torch.allclose(points["x_hat"], points["x"] / (2.0 + points["x"].abs()), atol=1e-6)

    scaled = cifar_resnet(depth=8, norm_layer=DynamicScale)
    layers, recorder = recorded(scaled, (torch.randn(2, 3, 32, 32), torch.tensor([0, 1])),
                                evaluator.classification())
    for probe in recorder.probes:
        points = layers[probe.name]
        assert torch.allclose(points["x_hat"], probe.module.alpha * points["x"], atol=1e-6)

    unshifted = cifar_resnet(depth=8, norm_layer=partial(DynamicTanh, bias=False))
    layers, recorder = recorded(unshifted, (torch.randn(2, 3, 32, 32), torch.tensor([0, 1])),
                                evaluator.classification())
    for probe in recorder.probes:
        points, c = layers[probe.name], layers[probe.name]["channel"]
        assert torch.allclose(points["y"], probe.module.weight[c] * points["x_hat"], atol=1e-5)

    resnet = cifar_resnet(depth=8, norm_layer=nn.Identity)
    layers, _ = recorded(resnet, (torch.randn(2, 3, 32, 32), torch.tensor([0, 1])),
                         evaluator.classification())
    for (name, points), width in zip(layers.items(), (16, 32, 64)):
        assert torch.equal(points["x_hat"], points["x"]) and torch.equal(points["y"], points["x"])
        assert points["channel"].max().item() == width - 1, name  # channels on axis 1
