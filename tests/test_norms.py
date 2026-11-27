"""Tests for the element-wise replacements of normalization layers."""

from functools import partial
from pathlib import Path

import pytest
import torch
from torch import nn

from nstudy.core import evaluator, norm_io
from nstudy.core.model import Transformer, cifar_resnet, imagenet_resnet
from nstudy.core.norms import (DynamicScale, DynamicSoftsign, DynamicTanh, ElementwiseNorm,
                               LayerNorm2d, hadamard_rotation)
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
    z = 0.5 * x
    assert torch.allclose(layer(x), layer.weight * z / (1 + z.abs()) + layer.bias)
    with torch.no_grad():
        layer.alpha.fill_(3.0)  # alpha multiplies, as in DyT
    z = 3.0 * x
    assert torch.allclose(layer(x), layer.weight * z / (1 + z.abs()) + layer.bias)


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


@pytest.mark.parametrize("norm", ["dyt", "dyt_channel", "dysoftsign", "dyscale", "none", "batchnorm", "layernorm",
                                  "layernorm2d"])
@pytest.mark.parametrize("experiment", ["cifar10_resnet", "wmt14_transformer_base"])
def test_norm_option_replaces_the_norm_layer_of_any_experiment(norm, experiment):
    hydra = pytest.importorskip("hydra")
    from hydra.utils import instantiate

    if (norm, experiment) in {("batchnorm", "wmt14_transformer_base"),
                              ("layernorm2d", "wmt14_transformer_base"),
                              ("layernorm", "cifar10_resnet")}:
        pytest.skip("BatchNorm2d and LayerNorm2d need images; LayerNorm here normalizes tokens")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=[f"+experiment={experiment}",
                                                             f"norm={norm}"])
    expected = {"dyt": DynamicTanh, "dyt_channel": DynamicTanh, "dysoftsign": DynamicSoftsign, "dyscale": DynamicScale,
                "none": nn.Identity,
                "batchnorm": nn.BatchNorm2d, "layernorm": nn.LayerNorm,
                "layernorm2d": LayerNorm2d}[norm]
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


def test_layernorm2d_normalizes_each_image_over_channels_and_positions():
    torch.manual_seed(0)
    x = torch.randn(4, 16, 8, 8) * 3 + 2
    out = LayerNorm2d(16)(x)
    flat = out.flatten(1)
    assert torch.allclose(flat.mean(1), torch.zeros(4), atol=1e-5)
    assert torch.allclose(flat.var(1, unbiased=False), torch.ones(4), atol=1e-3)
    assert LayerNorm2d(16).weight.shape == (16,)
    plain = cifar_resnet(20, norm_layer=LayerNorm2d, residual=False)
    assert plain(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_dyt_channel_option_learns_one_alpha_per_channel():
    hydra = pytest.importorskip("hydra")
    from hydra.utils import instantiate

    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        scalar, channel = (hydra.compose(config_name="config", overrides=[
            "+experiment=cifar10_resnet", f"norm={norm}"]) for norm in ("dyt", "dyt_channel"))
    assert instantiate(scalar.model.norm_layer)(16).alpha.shape == (1,)
    layer = instantiate(channel.model.norm_layer)(16)
    assert layer.alpha.shape == layer.weight.shape == layer.bias.shape == (16,)
    assert torch.all(layer.alpha == 0.5)


def test_dyt_dropout_acts_on_the_output_in_training_only():
    torch.manual_seed(0)
    plain, dropped = DynamicTanh(16), DynamicTanh(16, dropout=0.3)
    dropped.load_state_dict(plain.state_dict())  # no new parameters or buffers
    with torch.no_grad():
        plain.bias.fill_(0.5)
        dropped.bias.fill_(0.5)
    x = torch.randn(64, 16, 8, 8)
    assert torch.equal(dropped.eval()(x), plain.eval()(x))
    out, reference = dropped.train()(x), plain(x)
    zeroed = out == 0
    assert 0.27 < zeroed.float().mean().item() < 0.33  # beta is zeroed too: after the shift
    assert torch.allclose(out[~zeroed], reference[~zeroed] / 0.7)


def test_dropout_option_reaches_the_dyt_layers():
    hydra = pytest.importorskip("hydra")
    from hydra.utils import instantiate

    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        default = hydra.compose(config_name="config", overrides=[
            "+experiment=cifar10_resnet", "norm=dyt_channel"])
        dropped = hydra.compose(config_name="config", overrides=[
            "+experiment=cifar10_resnet", "norm=dyt_channel", "model.norm_layer.dropout=0.1"])
    assert isinstance(instantiate(default.model.norm_layer)(16).dropout, nn.Identity)
    layer = instantiate(dropped.model.norm_layer)(16)
    assert isinstance(layer.dropout, nn.Dropout) and layer.dropout.p == 0.1
    assert layer.alpha.shape == (16,)


def test_dyt_channel_dropout_zeroes_whole_channels():
    torch.manual_seed(0)
    plain, dropped = DynamicTanh(64), DynamicTanh(64, dropout=0.25, dropout_channels=True)
    x = torch.randn(32, 64, 8, 8)
    assert torch.equal(dropped.eval()(x), plain.eval()(x))
    out, reference = dropped.train()(x), plain(x)
    zeroed = (out == 0).all(dim=(2, 3))  # (image, channel) maps that are all zero
    assert 0.2 < zeroed.float().mean().item() < 0.3
    kept = ~zeroed
    assert torch.allclose(out[kept], reference[kept] / 0.75)
    with pytest.raises(ValueError, match="dropout_channels"):
        dropped.train()(torch.randn(2, 5, 64))  # Transformer tokens


def test_channel_dropout_option_reaches_the_dyt_layers():
    hydra = pytest.importorskip("hydra")
    from hydra.utils import instantiate

    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=[
            "+experiment=cifar10_resnet", "norm=dyt_channel", "model.norm_layer.dropout=0.05",
            "model.norm_layer.dropout_channels=true"])
    layer = instantiate(cfg.model.norm_layer)(16)
    assert isinstance(layer.dropout, nn.Dropout2d) and layer.dropout.p == 0.05


@pytest.mark.parametrize("norm, cls", [("dyt", DynamicTanh), ("dysoftsign", DynamicSoftsign),
                                       ("dyscale", DynamicScale)])
@pytest.mark.parametrize("channels", [False, True])
def test_every_elementwise_norm_takes_both_dropouts(norm, cls, channels):
    hydra = pytest.importorskip("hydra")
    from hydra.utils import instantiate

    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=[
            "+experiment=cifar10_resnet", f"norm={norm}", "model.norm_layer.dropout=0.1",
            f"model.norm_layer.dropout_channels={str(channels).lower()}"])
    layer = instantiate(cfg.model.norm_layer)(16)
    assert type(layer) is cls
    assert type(layer.dropout) is (nn.Dropout2d if channels else nn.Dropout)
    assert layer.dropout.p == 0.1


@pytest.mark.parametrize("n", [1, 2, 16, 64, 512])
def test_hadamard_rotation_is_orthogonal_with_entries_of_one_size(n):
    torch.manual_seed(0)
    rotation = hadamard_rotation(n)
    assert torch.allclose(rotation @ rotation.T, torch.eye(n), atol=1e-6)
    assert torch.allclose(rotation.abs(), torch.full((n, n), n**-0.5))
    with pytest.raises(ValueError, match="power-of-2"):
        hadamard_rotation(48)


@pytest.mark.parametrize("cls", [DynamicTanh, DynamicSoftsign, DynamicScale])
def test_hadamard_rotates_the_channels_between_f_and_gamma(cls):
    torch.manual_seed(0)
    layer = randomized(cls(16, hadamard=True, per_channel_alpha=True))
    plain = cls(16, per_channel_alpha=True)
    plain.load_state_dict({k: v for k, v in layer.state_dict().items() if k != "rotation"})
    for x, axis in ((torch.randn(2, 16, 4, 4), 1), (torch.randn(2, 3, 16), -1)):
        f = plain.squash(x, plain.per_channel(plain.alpha, x))
        rotated = (f.movedim(axis, -1) @ layer.rotation.T).movedim(-1, axis)
        expected = rotated * plain.per_channel(plain.weight, x) + plain.per_channel(plain.bias, x)
        assert torch.allclose(layer(x), expected, atol=1e-6)
    assert "rotation" in dict(layer.named_buffers())
    assert not any(p is layer.rotation for p in layer.parameters())  # fixed, not trained


def test_hadamard_rotations_differ_between_layers_and_follow_the_seed():
    torch.manual_seed(0)
    first, second = DynamicTanh(16, hadamard=True), DynamicTanh(16, hadamard=True)
    torch.manual_seed(0)
    again = DynamicTanh(16, hadamard=True)
    assert not torch.equal(first.rotation, second.rotation)
    assert torch.equal(first.rotation, again.rotation)


def test_scalar_alpha_dyscale_rotation_folds_into_the_conv_before_it():
    """gamma * R (alpha * conv(x)) + beta is gamma * conv'(x) + beta with conv' = alpha R W."""
    torch.manual_seed(0)
    conv = nn.Conv2d(8, 16, 3, padding=1, bias=False)
    layer = randomized(DynamicScale(16, hadamard=True))
    x = torch.randn(2, 8, 6, 6)
    folded = nn.Conv2d(8, 16, 3, padding=1, bias=False)
    with torch.no_grad():
        folded.weight.copy_(layer.alpha * torch.einsum("oc,cikl->oikl", layer.rotation, conv.weight))
    expected = folded(x) * layer.weight.view(1, -1, 1, 1) + layer.bias.view(1, -1, 1, 1)
    assert torch.allclose(layer(conv(x)), expected, atol=1e-5)


def test_hadamard_option_reaches_every_elementwise_norm():
    hydra = pytest.importorskip("hydra")
    from hydra.utils import instantiate

    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        for norm in ("dyt", "dyt_channel", "dysoftsign", "dyscale"):
            cfg = hydra.compose(config_name="config", overrides=[
                "+experiment=cifar10_resnet", f"norm={norm}", "model.norm_layer.hadamard=true",
                "model.norm_layer.dropout=0.05", "model.norm_layer.dropout_channels=true"])
            layer = instantiate(cfg.model.norm_layer)(32)
            assert layer.hadamard and layer.rotation.shape == (32, 32)
            assert isinstance(layer.dropout, nn.Dropout2d)


@pytest.mark.parametrize("cls", [DynamicTanh, DynamicSoftsign])
def test_fixed_alpha_is_a_buffer_kept_at_its_initial_value(cls):
    layer = cls(16, alpha_init=0.3, learn_alpha=False)
    assert [name for name, _ in layer.named_parameters()] == ["weight", "bias"]
    assert "alpha" in layer.state_dict() and layer.alpha.item() == pytest.approx(0.3)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1, weight_decay=0.1)
    layer(torch.randn(2, 16, 4, 4)).square().sum().backward()
    optimizer.step()
    assert layer.alpha.item() == pytest.approx(0.3) and layer.weight.grad is not None


def test_dyscale_without_alpha_is_gamma_x_plus_beta_with_gamma_from_one_over_sqrt_d():
    layer = DynamicScale(16, learn_alpha=False)
    assert [name for name, _ in layer.named_parameters()] == ["weight", "bias"]
    assert torch.all(layer.alpha == 1) and torch.allclose(layer.weight, torch.full((16,), 0.25))
    x = torch.randn(2, 16, 4, 4)
    assert torch.allclose(layer(x), x / 4)
    assert torch.allclose(DynamicScale(16, alpha_init=0.3, learn_alpha=False).weight, torch.full((16,), 0.3))
    # gamma's gradient has no factor alpha: the same as that of alpha * gamma with alpha learned
    (grad,) = torch.autograd.grad(layer(x).sum(), layer.weight)
    assert torch.allclose(grad, x.sum(dim=(0, 2, 3)))


def test_dyscale_without_alpha_starts_gamma_at_one_over_sqrt_fan_in_in_a_resnet():
    model = cifar_resnet(depth=8, norm_layer=partial(DynamicScale, learn_alpha=False))
    layers = {name: m for name, m in model.named_modules() if isinstance(m, DynamicScale)}
    assert torch.allclose(layers["bn1"].weight, torch.full((16,), 27 ** -0.5))
    assert torch.allclose(layers["layer2.0.bn2"].weight, torch.full((32,), 288 ** -0.5))
    assert all(torch.all(m.alpha == 1) for m in layers.values())
    assert not any("alpha" in name for name, _ in model.named_parameters())


@pytest.mark.parametrize("norm", ["dyt", "dysoftsign", "dyscale"])
def test_learn_alpha_option_reaches_the_layer(norm):
    import hydra
    from hydra.utils import instantiate

    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=[f"norm={norm}", "model.norm_layer.learn_alpha=false"])
    layer = instantiate(cfg.model.norm_layer)(16)
    assert not layer.learn_alpha and "alpha" not in dict(layer.named_parameters())
