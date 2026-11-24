"""Tests for recording normalization layers' input and output before scale and shift."""

from functools import partial
from pathlib import Path

import pytest
import torch
from torch import nn

from nstudy.core import evaluator, norm_io
from nstudy.core.model import Transformer, cifar_resnet
from nstudy.core.probes import model_probes

CPU = torch.device("cpu")
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
EVERYTHING = 10**9  # points per batch: keep every element


def trained_looking_resnet() -> nn.Module:
    """ResNet-8 whose norms have non-trivial running statistics, scales and shifts."""
    torch.manual_seed(0)
    model = cifar_resnet(depth=8, num_classes=10, shortcut="A", norm_layer=partial(nn.BatchNorm2d))
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.uniform_(-1, 1)
                module.running_var.uniform_(0.5, 2)
                module.weight.uniform_(0.5, 2)
                module.bias.uniform_(-1, 1)
    return model


def images(count: int = 6):
    return torch.randn(count, 3, 32, 32), torch.randint(0, 10, (count,))


def record(model, batches, mode="eval", points=EVERYTHING, examples=10**6):
    recorder = norm_io.NormIORecorder(model, model_probes(model), points, mode=mode)
    recorder.run(batches, evaluator.classification() if not isinstance(model, Transformer)
                 else evaluator.translation(), CPU, examples)
    recorder.remove()
    return {name: r.tensors() for name, r in recorder.records.items()}, recorder


def test_unit_affine_copy_gives_the_normalized_input():
    norm = trained_looking_resnet().layer1[0].bn1.eval()
    x = torch.randn(4, 16, 5, 5)
    expected = (x - norm.running_mean[:, None, None]) / torch.sqrt(norm.running_var[:, None, None] + norm.eps)
    assert torch.allclose(norm_io.unit_affine(norm)(x), expected, atol=1e-5)
    assert not torch.equal(norm.weight, torch.ones_like(norm.weight))  # the original is unchanged


def test_eval_mode_batchnorm_normalizes_with_running_statistics():
    model = trained_looking_resnet()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    layers, recorder = record(model, [images()])
    probes = {p.name: p for p in model_probes(model)}
    assert list(layers) == ["stage1.block1.layer2", "stage2.block1.layer4", "stage3.block1.layer6"]
    assert [recorder.records[n].position for n in layers] == ["early", "mid", "late"]
    for name, points in layers.items():
        norm = probes[name].module
        c = points["channel"]
        std = torch.sqrt(norm.running_var[c] + norm.eps)
        assert torch.allclose(points["x_hat"], (points["x"] - norm.running_mean[c]) / std, atol=1e-4)
        assert torch.allclose(points["y"], norm.weight[c] * points["x_hat"] + norm.bias[c], atol=1e-4)
    # ResNet-8 stage1.block1 bn1 output: 6 images x 16 channels x 32 x 32, all kept.
    assert layers["stage1.block1.layer2"]["x"].numel() == 6 * 16 * 32 * 32
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_train_mode_batchnorm_uses_batch_statistics_and_changes_nothing():
    model = trained_looking_resnet()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    layers, _ = record(model, [images(8)], mode="train")
    points = layers["stage1.block1.layer2"]
    for channel in range(16):
        x_hat = points["x_hat"][points["channel"] == channel]
        assert abs(x_hat.mean().item()) < 1e-4
        assert x_hat.var(unbiased=False).item() == pytest.approx(1.0, abs=1e-3)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key
    assert all(module.training for module in model.modules())  # train mode, as before


def test_examples_are_counted_in_whole_batches_and_points_are_capped():
    model = trained_looking_resnet()
    batches = [images(4) for _ in range(5)]
    _, recorder = record(model, batches, points=100, examples=10)
    assert recorder.examples == 12  # 3 batches of 4
    assert all(r.tensors()["x"].numel() == 300 for r in recorder.records.values())
    assert norm_io.per_batch_quota(200_000, examples=512, batch_examples=128) == 50_000


def test_transformer_layernorm_points_leave_out_padding():
    torch.manual_seed(0)
    model = Transformer(vocab_size=30, d_model=16, num_heads=2, num_layers=3, d_ff=32).eval()
    src = torch.tensor([[5, 6, 7, 0], [8, 9, 0, 0]])
    tgt = torch.tensor([[3, 4, 0], [3, 5, 6]])
    batch = {"src": src, "tgt_in": tgt, "tgt_out": tgt}
    layers, recorder = record(model, [batch])

    assert set(layers) == {f"{side}.layer{i}.norm{n}" for side in ("encoder", "decoder")
                           for i in (1, 2, 3) for n in ((1, 2) if side == "encoder" else (1, 2, 3))}
    assert recorder.records["encoder.layer1.norm1"].position == "early"
    assert recorder.records["decoder.layer3.norm3"].position == "late"
    for name, points in layers.items():
        # 5 non-padding tokens on each side (3 + 2 source, 2 + 3 target), 16 channels each.
        assert points["x"].numel() == 5 * 16, name
        # Every token is normalized to mean 0 and variance 1 over its 16 channels.
        assert abs(points["x_hat"].mean().item()) < 1e-4
        assert points["x_hat"].pow(2).mean().item() == pytest.approx(1.0, abs=1e-3)
        norm = next(p.module for p in recorder.probes if p.name == name)
        c = points["channel"]
        assert torch.allclose(points["y"], norm.weight[c] * points["x_hat"] + norm.bias[c], atol=1e-5)


def test_points_are_saved_and_plotted(tmp_path):
    pytest.importorskip("matplotlib")
    model = trained_looking_resnet()
    _, recorder = record(model, [images()], points=500)
    norm_io.save(recorder.records, tmp_path / "points.pt", {"mode": "eval"})
    saved = torch.load(tmp_path / "points.pt")
    assert saved["metadata"] == {"mode": "eval"}
    assert saved["layers"]["stage2.block1.layer4"]["position"] == "mid"
    assert saved["layers"]["stage2.block1.layer4"]["x"].numel() == 500
    norm_io.plot(saved["layers"], tmp_path / "norm_io.png", title="test")
    assert (tmp_path / "norm_io.png").stat().st_size > 0


def test_config_needs_only_the_run_directory():
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="norm_io", overrides=["run_dir=outputs/x"])
    assert (cfg.split, cfg.mode, cfg.examples) == ("val", "eval", 512)
    assert cfg.mode in norm_io.MODES
