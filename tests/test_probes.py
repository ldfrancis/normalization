"""Tests for the layer probes: statistics of chosen layers during training.

ResNet: in an early, a middle and a late block, the input of the ReLU after
``conv1`` (the output of ``bn1``) and the weights of ``conv1``.
Transformer: in an early, a middle and a late layer of the encoder and of the
decoder, the output of every normalization layer.
"""

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nstudy.core import evaluator
from nstudy.core.constants import PAD_ID
from nstudy.core.model import Transformer, cifar_resnet, imagenet_resnet
from nstudy.core.probes import (
    ProbeRecorder,
    default_blocks,
    percentiles,
    resnet_probes,
    transformer_probes,
)
from nstudy.core.trainer import Trainer, step_schedule

CPU = torch.device("cpu")
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


# ---------------------------------------------------------------------------
# Which layers are watched
# ---------------------------------------------------------------------------


def test_default_blocks_are_first_middle_and_last():
    assert default_blocks(27) == [0, 13, 26]
    assert default_blocks(16) == [0, 7, 15]
    assert default_blocks(6) == [0, 2, 5]
    assert default_blocks(1) == [0]


def test_cifar_resnet_probes_watch_the_relu_input_and_conv1_of_three_blocks():
    model = cifar_resnet(56)  # 27 blocks: 9 per stage
    probes = resnet_probes(model)
    assert [probe.name for probe in probes] == ["layer1.0", "layer2.4", "layer3.8"]
    blocks = [model.layer1[0], model.layer2[4], model.layer3[8]]
    for probe, block in zip(probes, blocks):
        assert probe.module is block.bn1  # its output is the ReLU's input
        assert probe.weight is block.conv1.weight
        assert probe.activation == "relu_input"


def test_imagenet_resnet_probes_span_the_stages():
    probes = resnet_probes(imagenet_resnet(50))  # 3 + 4 + 6 + 3 blocks
    assert [probe.name for probe in probes] == ["layer1.0", "layer3.0", "layer4.2"]


def test_probes_can_watch_chosen_blocks():
    probes = resnet_probes(cifar_resnet(20), blocks=[1, -1])
    assert [probe.name for probe in probes] == ["layer1.1", "layer3.2"]


def test_transformer_probes_watch_every_norm_of_three_encoder_and_decoder_layers():
    model = Transformer(50, d_model=16, num_heads=2, num_layers=6, d_ff=32)
    names = [probe.name for probe in transformer_probes(model)]
    assert names == [
        "encoder.0.norm1", "encoder.0.norm2",
        "encoder.2.norm1", "encoder.2.norm2",
        "encoder.5.norm1", "encoder.5.norm2",
        "decoder.0.norm1", "decoder.0.norm2", "decoder.0.norm3",
        "decoder.2.norm1", "decoder.2.norm2", "decoder.2.norm3",
        "decoder.5.norm1", "decoder.5.norm2", "decoder.5.norm3",
    ]
    probes = transformer_probes(model)
    assert probes[0].module is model.encoder[0].norm1
    assert all(probe.activation == "output" for probe in probes)
    # The last linear of the sub-layer each norm wraps.
    encoder, decoder = model.encoder[2], model.decoder[5]
    weights = {probe.name: (probe.weight, probe.weight_name) for probe in probes}
    assert weights["encoder.2.norm1"] == (encoder.self_attn.out_proj.weight, "self_attn_out_proj_weight")
    assert weights["encoder.2.norm2"] == (encoder.linear2.weight, "linear2_weight")
    assert weights["decoder.5.norm1"] == (decoder.self_attn.out_proj.weight, "self_attn_out_proj_weight")
    assert weights["decoder.5.norm2"] == (decoder.cross_attn.out_proj.weight, "cross_attn_out_proj_weight")
    assert weights["decoder.5.norm3"] == (decoder.linear2.weight, "linear2_weight")


# ---------------------------------------------------------------------------
# What is measured
# ---------------------------------------------------------------------------


def test_percentiles_of_known_values():
    values = torch.arange(101.0)[torch.randperm(101)]
    assert percentiles(values, (15, 50, 85)) == [15.0, 50.0, 85.0]


def reference_relu_inputs(model: nn.Module, images, labels, blocks):
    """bn1 outputs and their gradients, from a copy whose ReLUs are not in place."""
    model = copy.deepcopy(model).train()
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    outputs = {}

    def keep(name):
        def hook(module, inputs, output):
            output.retain_grad()
            outputs[name] = output
        return hook

    for name, block in blocks(model).items():
        block.bn1.register_forward_hook(keep(name))
    nn.functional.cross_entropy(model(images), labels).backward()
    return outputs, model


def resnet_and_batch(seed: int = 0):
    torch.manual_seed(seed)
    model = cifar_resnet(20)
    batch = (torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)))
    return model, batch


def test_resnet_statistics_match_a_direct_computation():
    model, (images, labels) = resnet_and_batch()
    blocks = lambda m: {"layer1.0": m.layer1[0], "layer2.1": m.layer2[1], "layer3.2": m.layer3[2]}  # noqa: E731
    expected, reference = reference_relu_inputs(model, images, labels, blocks)

    recorder = ProbeRecorder(model, resnet_probes(model), batch=(images, labels))
    stats, histograms = recorder.run(evaluator.classification(), CPU)

    for name, output in expected.items():
        weight = blocks(reference)[name].conv1.weight
        # Both sizes of every tensor: the L2 norm, and the root mean square
        # (the typical size of one element, whatever the tensor's size).
        rms = lambda t: t.detach().pow(2).mean().sqrt().item()  # noqa: E731
        l2 = lambda t: t.detach().norm().item()  # noqa: E731
        for key, tensor in [("relu_input", output), ("relu_input_grad", output.grad),
                            ("conv1_weight", weight), ("conv1_weight_grad", weight.grad)]:
            assert stats[f"{name}/{key}/rms"] == pytest.approx(rms(tensor), rel=1e-5)
            assert stats[f"{name}/{key}/l2"] == pytest.approx(l2(tensor), rel=1e-5)
        # The ReLU input has negative values: it was recorded before the ReLU.
        p15, p50, p85 = (stats[f"{name}/relu_input/p{q}"] for q in (15, 50, 85))
        assert p15 < 0 < p85 and p15 <= p50 <= p85
        assert [p15, p50, p85] == pytest.approx(percentiles(output.detach(), (15, 50, 85)))
        assert stats[f"{name}/conv1_weight/p50"] == pytest.approx(
            percentiles(weight.detach(), (50,))[0]
        )
        assert sum(histograms[f"{name}/relu_input"]["counts"]) == output.numel()
        assert sum(histograms[f"{name}/conv1_weight"]["counts"]) == weight.numel()


def test_a_probe_pass_leaves_no_trace():
    """No gradients, no change to BatchNorm statistics, mode or random numbers."""
    model, batch = resnet_and_batch()
    model.eval()
    buffers = [buffer.clone() for buffer in model.buffers()]
    rng = torch.get_rng_state()
    recorder = ProbeRecorder(model, resnet_probes(model), batch=batch)
    recorder.run(evaluator.classification(), CPU)

    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(buffers, model.buffers()))
    assert not model.training
    assert torch.equal(torch.get_rng_state(), rng)
    # Outside a probe pass the hooks do nothing.
    out = model(batch[0])
    assert recorder.activations == {}
    assert torch.equal(out, model(batch[0]))


def test_the_same_weights_give_the_same_statistics():
    """The batch and the dropout masks are fixed, so only training changes the numbers."""
    torch.manual_seed(0)
    model = Transformer(30, d_model=16, num_heads=2, num_layers=3, d_ff=32, dropout=0.3)
    batch = {
        "src": torch.randint(3, 30, (4, 7)),
        "tgt_in": torch.randint(3, 30, (4, 6)),
        "tgt_out": torch.randint(3, 30, (4, 6)),
    }
    recorder = ProbeRecorder(model, transformer_probes(model), batch=batch)
    task = evaluator.translation(label_smoothing=0.1)
    first = recorder.run(task, CPU)
    torch.rand(100)  # whatever random numbers training draws in between
    assert recorder.run(task, CPU) == first


def test_transformer_statistics_leave_out_padding():
    torch.manual_seed(0)
    model = Transformer(30, d_model=16, num_heads=2, num_layers=3, d_ff=32, dropout=0.0)
    batch = {
        "src": torch.tensor([[5, 6, 7, 2], [8, 2, PAD_ID, PAD_ID]]),  # 6 real source tokens
        "tgt_in": torch.tensor([[1, 9, 10], [1, 11, PAD_ID]]),  # 5 real target tokens
        "tgt_out": torch.tensor([[9, 10, 2], [11, 2, PAD_ID]]),
    }
    recorder = ProbeRecorder(model, transformer_probes(model), batch=batch)
    stats, histograms = recorder.run(evaluator.translation(), CPU)
    assert sum(histograms["encoder.0.norm1/output"]["counts"]) == 6 * 16
    assert sum(histograms["decoder.2.norm3/output"]["counts"]) == 5 * 16
    assert {"encoder.0.norm1/output/rms", "encoder.0.norm1/output_grad/rms",
            "encoder.0.norm1/output/l2", "encoder.0.norm1/output_grad/l2",
            "decoder.1.norm2/output/p85"} <= set(stats)
    # Weights have no padding: all their elements are measured.
    weight = model.decoder[2].linear2.weight
    assert sum(histograms["decoder.2.norm3/linear2_weight"]["counts"]) == weight.numel()
    assert {"encoder.0.norm1/self_attn_out_proj_weight/l2",
            "encoder.0.norm1/self_attn_out_proj_weight_grad/l2",
            "decoder.1.norm2/cross_attn_out_proj_weight/rms"} <= set(stats)


# ---------------------------------------------------------------------------
# During training
# ---------------------------------------------------------------------------


class ListLogger:
    def __init__(self):
        self.rows, self.histograms = [], []

    def log(self, row):
        self.rows.append(dict(row))

    def log_histograms(self, step, histograms):
        self.histograms.append((step, set(histograms)))


def train_resnet(tmp_path, probes: bool, logger=None, lr=0.1):
    torch.manual_seed(0)
    data = TensorDataset(torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,)))
    model = cifar_resnet(20)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    recorder = None
    if probes:
        probe_batch = (data.tensors[0][:8], data.tensors[1][:8])
        recorder = ProbeRecorder(model, resnet_probes(model), probe_batch,
                                 histogram_every=2, stats_every=3)
    trainer = Trainer(
        model=model,
        loaders={"train": DataLoader(data, batch_size=16, shuffle=True)},
        task=evaluator.classification(),
        optimizer=optimizer,
        scheduler=step_schedule(optimizer, milestones=[]),
        max_steps=6,
        output_dir=tmp_path,
        eval_every=3,
        log_every=3,
        device=CPU,
        probes=recorder,
        logger=logger,
    )
    trainer.fit()
    return trainer


def read_rows(path: Path, name: str = "metrics.jsonl") -> list[dict]:
    return [json.loads(line) for line in (path / name).read_text().splitlines()]


def test_trainer_records_probe_statistics_and_histograms_on_their_schedules(tmp_path):
    logger = ListLogger()
    train_resnet(tmp_path, probes=True, logger=logger)
    probe_rows = [row for row in read_rows(tmp_path) if row["split"] == "probe"]
    assert [row["step"] for row in probe_rows] == [3, 6]
    assert {"layer2.1/conv1_weight_grad/rms", "layer2.1/conv1_weight_grad/l2"} <= set(probe_rows[0])

    histogram_rows = read_rows(tmp_path, "histograms.jsonl")
    assert [row["step"] for row in histogram_rows] == [2, 4, 6]
    assert set(histogram_rows[0]["histograms"]) == {
        f"{block}/{quantity}"
        for block in ("layer1.0", "layer2.1", "layer3.2")
        for quantity in ("relu_input", "conv1_weight")
    }
    assert [step for step, _ in logger.histograms] == [2, 4, 6]


def test_probe_statistics_change_only_when_the_weights_do(tmp_path):
    """With a learning rate of 0, every probe step must report the same numbers,
    although training sees different shuffled batches."""
    train_resnet(tmp_path, probes=True, lr=0.0)
    first, second = (
        {key: value for key, value in row.items() if key != "step"}
        for row in read_rows(tmp_path) if row["split"] == "probe"
    )
    assert first == second


def test_probes_do_not_change_training(tmp_path):
    with_probes = train_resnet(tmp_path / "with", probes=True)
    without = train_resnet(tmp_path / "without", probes=False)
    for ours, reference in zip(with_probes.model.parameters(), without.model.parameters()):
        assert torch.equal(ours, reference)
    for ours, reference in zip(with_probes.model.buffers(), without.model.buffers()):
        assert torch.equal(ours, reference)


def test_config_has_probe_settings():
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=["+experiment=imagenet_resnet"])
    assert cfg.probe.enabled
    assert cfg.probe.blocks is None  # first, middle and last
    assert cfg.probe.stats_every == cfg.trainer.eval_every
    assert cfg.probe.histogram_every == cfg.trainer.eval_every
