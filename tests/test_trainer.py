"""Tests for the trainer and the training configs.

The hyperparameters are checked against He et al. (2016) for the ResNets and
Vaswani et al. (2017) for the Transformer.
"""

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nstudy.core import evaluator
from nstudy.core.constants import PAD_ID
from nstudy.core.data import tokenizer_path
from nstudy.core.model import Transformer, cifar_resnet
from nstudy.core.trainer import Trainer, noam_lr, noam_schedule, step_factor, step_schedule

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Learning-rate schedules
# ---------------------------------------------------------------------------


def test_cifar_schedule_divides_by_10_at_32k_and_48k():
    milestones = [32_000, 48_000]
    assert step_factor(0, milestones) == 1.0
    assert step_factor(31_999, milestones) == 1.0
    assert step_factor(32_000, milestones) == pytest.approx(0.1)
    assert step_factor(47_999, milestones) == pytest.approx(0.1)
    assert step_factor(48_000, milestones) == pytest.approx(0.01)
    assert step_factor(63_999, milestones) == pytest.approx(0.01)


def test_resnet110_warmup_starts_ten_times_lower_for_400_steps():
    milestones = [32_000, 48_000]
    assert step_factor(0, milestones, warmup_steps=400) == 0.1
    assert step_factor(399, milestones, warmup_steps=400) == 0.1
    assert step_factor(400, milestones, warmup_steps=400) == 1.0


def test_step_schedule_drives_the_optimizer():
    optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
    scheduler = step_schedule(optimizer, milestones=[3, 5])
    rates = []
    for _ in range(7):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates == pytest.approx([0.1, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001])


def test_noam_rate_follows_equation_3():
    d_model, warmup = 512, 4000
    # Linear warm-up: step * warmup^-1.5, scaled by d_model^-0.5.
    assert noam_lr(1, d_model, warmup) == pytest.approx(512**-0.5 * 4000**-1.5)
    assert noam_lr(2000, d_model, warmup) == pytest.approx(noam_lr(4000, d_model, warmup) / 2)
    # The peak is at the end of warm-up.
    peak = noam_lr(4000, d_model, warmup)
    assert peak == pytest.approx((512 * 4000) ** -0.5)
    assert peak == max(noam_lr(step, d_model, warmup) for step in range(1, 20_000))
    assert noam_lr(3999, d_model, warmup) < peak > noam_lr(4001, d_model, warmup)
    # Afterwards it decays with the inverse square root of the step.
    assert noam_lr(16_000, d_model, warmup) == pytest.approx(peak / 2)


def test_noam_schedule_drives_the_optimizer_from_step_1():
    optimizer = torch.optim.Adam(nn.Linear(2, 2).parameters(), lr=1.0)
    scheduler = noam_schedule(optimizer, d_model=512, warmup_steps=4000)
    for step in (1, 2, 3):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(noam_lr(step, 512, 4000))
        optimizer.step()
        scheduler.step()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def toy_data(count: int = 256, seed: int = 0) -> TensorDataset:
    """Linearly separable two-class data."""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(count, 8, generator=generator)
    return TensorDataset(inputs, (inputs[:, 0] > 0).long())


def toy_trainer(tmp_path, loaders, max_steps, lr=0.5, momentum=0.0, milestones=(), **kwargs):
    torch.manual_seed(0)
    model = nn.Linear(8, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    return Trainer(
        model=model,
        loaders=loaders,
        task=evaluator.classification(),
        optimizer=optimizer,
        scheduler=step_schedule(optimizer, milestones=list(milestones)),
        max_steps=max_steps,
        output_dir=tmp_path,
        device=CPU,
        **kwargs,
    )


def read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]


def test_trainer_learns_and_reports_held_out_metrics(tmp_path):
    loaders = {
        "train": DataLoader(toy_data(seed=0), batch_size=32, shuffle=True),
        "val": DataLoader(toy_data(seed=1), batch_size=64),
        "test": DataLoader(toy_data(seed=2), batch_size=64),
    }
    trainer = toy_trainer(tmp_path, loaders, max_steps=200, eval_every=50, log_every=25)
    before = evaluator.evaluate(trainer.model, loaders["test"], trainer.task, CPU)
    results = trainer.fit()

    assert set(results) == {"val", "test"}
    assert results["test"]["error"] < 5
    assert results["test"]["loss"] < before["loss"]
    assert trainer.model.training

    rows = read_metrics(tmp_path)
    steps = lambda split: [row["step"] for row in rows if row["split"] == split]  # noqa: E731
    assert steps("train") == list(range(25, 201, 25))
    assert steps("val") == [50, 100, 150, 200]
    assert steps("test") == [200]
    assert all(row["lr"] == 0.5 for row in rows if row["split"] == "train")
    # The training error is logged with the training loss, and falls too.
    train_rows = [row for row in rows if row["split"] == "train"]
    assert train_rows[0]["error"] > train_rows[-1]["error"]
    assert train_rows[-1]["error"] < 10


def test_training_error_is_the_error_on_the_training_batches_since_the_last_log(tmp_path):
    """With a learning rate of 0 the model never changes, so the logged training
    loss and error must equal those of the model on the whole training set."""
    dataset = toy_data(64)
    loaders = {"train": DataLoader(dataset, batch_size=16)}  # 4 batches per pass
    trainer = toy_trainer(tmp_path, loaders, max_steps=8, lr=0.0, log_every=4)
    expected = evaluator.evaluate(
        trainer.model, DataLoader(dataset, batch_size=64), trainer.task, CPU
    )
    assert 0 < expected["error"] < 100
    trainer.fit()
    rows = [row for row in read_metrics(tmp_path) if row["split"] == "train"]
    assert [row["step"] for row in rows] == [4, 8]
    for row in rows:  # each window of 4 steps is one pass over the data
        assert row["error"] == pytest.approx(expected["error"])
        assert row["loss"] == pytest.approx(expected["loss"])


def test_trainer_evaluates_the_chosen_splits_during_training(tmp_path):
    """The CIFAR figures of He et al. plot the test error throughout training."""
    loaders = {
        "train": DataLoader(toy_data(seed=0), batch_size=32, shuffle=True),
        "val": DataLoader(toy_data(seed=1), batch_size=64),
        "test": DataLoader(toy_data(seed=2), batch_size=64),
    }
    trainer = toy_trainer(tmp_path, loaders, max_steps=6, eval_every=2,
                          eval_splits=["val", "test"])
    trainer.fit()
    rows = read_metrics(tmp_path)
    for split in ("val", "test"):
        assert [row["step"] for row in rows if row["split"] == split] == [2, 4, 6]


@pytest.mark.parametrize("splits", [["train"], ["test"]])
def test_trainer_rejects_eval_splits_it_cannot_evaluate(tmp_path, splits):
    loaders = {"train": DataLoader(toy_data(64), batch_size=32),
               "val": DataLoader(toy_data(64), batch_size=32)}
    with pytest.raises(ValueError):
        toy_trainer(tmp_path, loaders, max_steps=1, eval_splits=splits)


def test_trainer_makes_exactly_max_steps_updates(tmp_path):
    loaders = {"train": DataLoader(toy_data(64), batch_size=32)}  # 2 batches per pass
    trainer = toy_trainer(tmp_path, loaders, max_steps=7)
    assert trainer.fit() == {}
    assert trainer.step == 7
    assert trainer.scheduler.last_epoch == 7


def test_accumulating_two_half_batches_equals_one_full_batch(tmp_path):
    dataset = toy_data(64)
    full = toy_trainer(tmp_path / "full", {"train": DataLoader(dataset, batch_size=32)}, max_steps=2)
    halves = toy_trainer(
        tmp_path / "halves", {"train": DataLoader(dataset, batch_size=16)}, max_steps=2, accumulate=2
    )
    full.fit()
    halves.fit()
    assert halves.step == 2
    for ours, reference in zip(halves.model.parameters(), full.model.parameters()):
        assert torch.allclose(ours, reference, atol=1e-6)


def test_resuming_from_a_checkpoint_continues_the_same_run(tmp_path):
    # One batch per pass, so the result does not depend on where the data restarts.
    loaders = {"train": DataLoader(toy_data(64), batch_size=64)}
    settings = dict(momentum=0.9, milestones=[4])

    straight = toy_trainer(tmp_path / "straight", loaders, max_steps=6, **settings)
    straight.fit()

    first = toy_trainer(tmp_path / "first", loaders, max_steps=3, **settings)
    first.fit()
    second = toy_trainer(tmp_path / "second", loaders, max_steps=6, **settings)
    second.load_checkpoint(first.checkpoint_path)
    assert second.step == 3
    second.fit()

    assert second.step == 6
    assert second.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    for ours, reference in zip(second.model.parameters(), straight.model.parameters()):
        assert torch.allclose(ours, reference, atol=1e-6)


def test_trainer_stops_when_the_loss_is_not_finite(tmp_path):
    loaders = {"train": DataLoader(toy_data(64), batch_size=32)}
    trainer = toy_trainer(tmp_path, loaders, max_steps=50, lr=float("inf"))
    with pytest.raises(FloatingPointError):
        trainer.fit()


def test_trainer_trains_a_cifar_resnet(tmp_path):
    torch.manual_seed(0)
    data = TensorDataset(torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,)))
    model = cifar_resnet(8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    trainer = Trainer(
        model=model,
        loaders={"train": DataLoader(data, batch_size=16), "test": DataLoader(data, batch_size=16)},
        task=evaluator.classification(),
        optimizer=optimizer,
        scheduler=step_schedule(optimizer, milestones=[]),
        max_steps=30,
        output_dir=tmp_path,
        device=CPU,
    )
    results = trainer.fit()
    assert results["test"]["error"] < 50  # memorizes 16 images; chance is 90


def test_trainer_trains_a_transformer_to_copy(tmp_path):
    """With label smoothing and the paper's optimizer, on a tiny copy task."""
    torch.manual_seed(0)
    tokens = torch.randint(3, 20, (64, 6))
    bos = torch.full((64, 1), 1)
    batch = {"src": tokens, "tgt_in": torch.cat([bos, tokens[:, :-1]], dim=1), "tgt_out": tokens}
    model = Transformer(20, d_model=32, num_heads=2, num_layers=1, d_ff=64, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    task = evaluator.translation(label_smoothing=0.1)
    trainer = Trainer(
        model=model,
        loaders={"train": [batch], "test": [batch]},
        task=task,
        optimizer=optimizer,
        scheduler=noam_schedule(optimizer, d_model=32, warmup_steps=50),
        max_steps=150,
        output_dir=tmp_path,
        device=CPU,
    )
    before = evaluator.evaluate(model, [batch], task, CPU)
    results = trainer.fit()
    assert results["test"]["loss"] < before["loss"] / 2
    # Translation is not classification: no training error is logged.
    (row,) = [row for row in read_metrics(tmp_path) if row["split"] == "train"]
    assert "error" not in row and {"loss", "lr", "throughput"} <= set(row)


# ---------------------------------------------------------------------------
# Hydra configs: the papers' hyperparameters
# ---------------------------------------------------------------------------


def compose(overrides: list[str]):
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return hydra.compose(config_name="config", overrides=overrides)


def build_optimizer(cfg):
    from hydra.utils import instantiate

    optimizer = instantiate(cfg.trainer.optimizer)(nn.Linear(2, 2).parameters())
    return optimizer, instantiate(cfg.trainer.scheduler)(optimizer)


def rate_at(cfg, step: int) -> float:
    """Learning rate used for update number ``step`` (from 0)."""
    optimizer, scheduler = build_optimizer(cfg)
    scheduler.last_epoch = step - 1
    optimizer.step()
    scheduler.step()
    return optimizer.param_groups[0]["lr"]


@pytest.mark.parametrize(
    "experiment, data, batch_size", [("cifar10_resnet", "cifar10", 128), ("cifar100_resnet", "cifar100", 128)]
)
def test_cifar_experiment_uses_the_resnet_paper_settings(experiment, data, batch_size):
    """Section 4.2: SGD, momentum 0.9, weight decay 1e-4, batch 128, 64k iterations."""
    from hydra.utils import instantiate

    cfg = compose([f"+experiment={experiment}"])
    assert cfg.data.name == data
    assert cfg.data.batch_size == batch_size
    assert cfg.model.depth == 56
    assert cfg.model.num_classes == (100 if data == "cifar100" else 10)
    assert cfg.trainer.max_steps == 64_000
    assert cfg.trainer.accumulate == 1
    assert list(cfg.trainer.eval_splits) == ["val", "test"]  # Figure 6 plots the test error

    optimizer, _ = build_optimizer(cfg)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["momentum"] == 0.9
    assert optimizer.defaults["weight_decay"] == 1e-4
    assert rate_at(cfg, 0) == pytest.approx(0.1)
    assert rate_at(cfg, 32_000) == pytest.approx(0.01)
    assert rate_at(cfg, 48_000) == pytest.approx(0.001)

    task = instantiate(cfg.trainer.task)
    assert task.forward is evaluator.classification_forward
    assert task.label_smoothing == 0.0
    assert task.report_error


def test_imagenet_experiment_uses_the_resnet_paper_settings():
    """Section 3.4: SGD, momentum 0.9, weight decay 1e-4, batch 256, lr 0.1 divided by 10."""
    cfg = compose(["+experiment=imagenet_resnet"])
    assert cfg.model._target_.endswith("imagenet_resnet")
    assert cfg.model.depth == 50
    assert cfg.data.batch_size == 256
    assert cfg.trainer.max_steps == 450_000  # 90 epochs of 5005 iterations
    assert list(cfg.trainer.eval_splits) == ["val"]  # Figure 4 plots the validation error

    optimizer, _ = build_optimizer(cfg)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["momentum"] == 0.9
    assert optimizer.defaults["weight_decay"] == 1e-4
    assert rate_at(cfg, 0) == pytest.approx(0.1)
    assert rate_at(cfg, 150_000) == pytest.approx(0.01)
    assert rate_at(cfg, 300_000) == pytest.approx(0.001)


@pytest.mark.parametrize(
    "experiment, d_model, dropout, max_steps",
    [("wmt14_transformer_base", 512, 0.1, 100_000), ("wmt14_transformer_big", 1024, 0.3, 300_000)],
)
def test_transformer_experiment_uses_the_transformer_paper_settings(
    experiment, d_model, dropout, max_steps
):
    """Section 5: Adam(0.9, 0.98, 1e-9), 4000 warm-up steps, label smoothing 0.1."""
    from hydra.utils import instantiate

    cfg = compose([f"+experiment={experiment}"])
    assert cfg.model.d_model == d_model
    assert cfg.model.dropout == dropout
    assert cfg.model.vocab_size == cfg.data.vocab_size == 37_000
    assert cfg.data.max_tokens == 25_000
    assert cfg.trainer.max_steps == max_steps
    assert list(cfg.trainer.eval_splits) == ["val"]

    optimizer, _ = build_optimizer(cfg)
    assert isinstance(optimizer, torch.optim.Adam)
    assert tuple(optimizer.defaults["betas"]) == (0.9, 0.98)
    assert optimizer.defaults["eps"] == 1e-9
    assert optimizer.defaults["weight_decay"] == 0
    assert rate_at(cfg, 0) == pytest.approx(noam_lr(1, d_model, 4000))
    assert rate_at(cfg, 3999) == pytest.approx((d_model * 4000) ** -0.5)

    task = instantiate(cfg.trainer.task)
    assert task.forward is evaluator.translation_forward
    assert task.label_smoothing == 0.1
    assert task.ignore_index == PAD_ID
    # Section 6.1: beam size 4, length penalty 0.6, at most input length + 50.
    translator = task.translator
    assert (translator.beam_size, translator.length_penalty) == (4, 0.6)
    assert translator.max_extra_length == 50
    expected_file = tokenizer_path(cfg.data.root, cfg.data.vocab_size)
    assert Path(translator.tokenizer_file) == expected_file
