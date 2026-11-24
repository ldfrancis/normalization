"""Tests for experiment tracking: what the trainer logs, and the wandb logger.

wandb runs in offline mode here, so nothing is sent over the network.
"""

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nstudy.core import evaluator
from nstudy.core.trainer import Trainer, step_schedule
from nstudy.core.tracking import WandbLogger, run_tags, to_wandb

CPU = torch.device("cpu")
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def toy_data(count: int = 64, seed: int = 0) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(count, 8, generator=generator)
    return TensorDataset(inputs, (inputs[:, 0] > 0).long())


class ListLogger:
    """Stands in for wandb: keeps every row it is given."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def log(self, row: dict) -> None:
        self.rows.append(dict(row))


def toy_trainer(tmp_path, loaders, max_steps, lr=0.5, **kwargs) -> Trainer:
    torch.manual_seed(0)
    model = nn.Linear(8, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return Trainer(
        model=model,
        loaders=loaders,
        task=evaluator.classification(),
        optimizer=optimizer,
        scheduler=step_schedule(optimizer, milestones=[]),
        max_steps=max_steps,
        output_dir=tmp_path,
        device=CPU,
        **kwargs,
    )


def read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]


# ---------------------------------------------------------------------------
# What the trainer logs
# ---------------------------------------------------------------------------


def test_logger_receives_every_row_written_to_metrics_jsonl(tmp_path):
    loaders = {
        "train": DataLoader(toy_data(), batch_size=32),
        "val": DataLoader(toy_data(seed=1), batch_size=32),
        "test": DataLoader(toy_data(seed=2), batch_size=32),
    }
    logger = ListLogger()
    trainer = toy_trainer(tmp_path, loaders, max_steps=4, eval_every=2, log_every=2,
                          logger=logger)
    trainer.fit()
    assert logger.rows == read_metrics(tmp_path)
    assert [(row["step"], row["split"]) for row in logger.rows] == [
        (2, "train"), (2, "val"), (4, "train"), (4, "val"), (4, "test")
    ]


class SummaryLogger(ListLogger):
    def __init__(self) -> None:
        super().__init__()
        self.summaries: dict = {}

    def summary(self, values: dict) -> None:
        self.summaries.update(values)


def test_final_model_is_evaluated_on_every_held_out_split(tmp_path):
    """Test is not watched during training (eval_splits=val) but is evaluated at the end."""
    loaders = {
        "train": DataLoader(toy_data(), batch_size=32),
        "val": DataLoader(toy_data(seed=1), batch_size=32),
        "test": DataLoader(toy_data(seed=2), batch_size=32),
    }
    logger = SummaryLogger()
    trainer = toy_trainer(tmp_path, loaders, max_steps=4, eval_every=2, eval_splits=["val"],
                          logger=logger)
    results = trainer.fit()

    final = json.loads((tmp_path / "final_metrics.json").read_text())
    assert final == {"step": 4, **results}
    assert set(results) == {"val", "test"}
    test_rows = [row for row in read_metrics(tmp_path) if row["split"] == "test"]
    assert [row["step"] for row in test_rows] == [4]
    assert results["test"] == evaluator.evaluate(trainer.module, loaders["test"],
                                                 evaluator.classification(), CPU)
    assert logger.summaries == {f"final/{split}/{key}": value
                                for split, metrics in results.items()
                                for key, value in metrics.items()}


def test_training_rows_report_progress(tmp_path):
    loaders = {"train": DataLoader(toy_data(64), batch_size=16)}  # 4 batches per pass
    trainer = toy_trainer(tmp_path, loaders, max_steps=12, log_every=6)
    trainer.fit()
    rows = read_metrics(tmp_path)
    assert [row["epoch"] for row in rows] == [1.5, 3.0]
    for row in rows:
        assert row["throughput"] > 0  # training images per second
        assert row["step_time"] > 0  # seconds per update
        assert {"grad_norm", "param_norm"} <= set(row)


def test_gradient_and_parameter_norms_are_those_of_the_logged_step(tmp_path):
    """With a learning rate of 0 and one batch per pass, every step has the same gradient."""
    dataset = toy_data(64)
    loaders = {"train": DataLoader(dataset, batch_size=64)}
    trainer = toy_trainer(tmp_path, loaders, max_steps=2, lr=0.0, log_every=2)

    inputs, labels = dataset.tensors
    trainer.model.zero_grad()
    nn.functional.cross_entropy(trainer.model(inputs), labels).backward()
    grad_norm = torch.cat([p.grad.flatten() for p in trainer.model.parameters()]).norm()
    param_norm = torch.cat([p.detach().flatten() for p in trainer.model.parameters()]).norm()

    trainer.fit()
    (row,) = read_metrics(tmp_path)
    assert row["grad_norm"] == pytest.approx(grad_norm.item(), rel=1e-5)
    assert row["grad_norm_mean"] == pytest.approx(grad_norm.item(), rel=1e-5)
    assert row["param_norm"] == pytest.approx(param_norm.item(), rel=1e-5)


def test_grad_norm_mean_averages_every_update_since_the_last_record(tmp_path):
    """grad_norm is the last update's norm; grad_norm_mean the mean over the window."""
    loaders = {"train": DataLoader(toy_data(64), batch_size=16)}  # a different batch each step
    trainer = toy_trainer(tmp_path, loaders, max_steps=8, log_every=4)
    norms = []
    trainer.optimizer.register_step_pre_hook(lambda *_: norms.append(
        torch.cat([p.grad.flatten() for p in trainer.module.parameters()]).norm().item()))
    trainer.fit()
    rows = read_metrics(tmp_path)
    for row, window in zip(rows, (norms[:4], norms[4:])):
        assert row["grad_norm"] == pytest.approx(window[-1], rel=1e-5)
        assert row["grad_norm_mean"] == pytest.approx(sum(window) / 4, rel=1e-5)
    assert len(set(norms)) > 1  # the windows really mix different gradients


def test_probed_weights_get_their_training_gradient_norms(tmp_path):
    """The early, middle and late conv1 weights of a ResNet, at every log_every."""
    from nstudy.core.model import cifar_resnet
    from nstudy.core.probes import ProbeRecorder, model_probes

    images = torch.randn(32, 3, 8, 8)
    labels = torch.randint(0, 10, (32,))
    loaders = {"train": DataLoader(TensorDataset(images, labels), batch_size=8)}
    torch.manual_seed(0)
    model = cifar_resnet(8)
    probes = ProbeRecorder(model, model_probes(model), batch=(images[:8], labels[:8]),
                           histogram_every=100, stats_every=100)
    norms: dict[str, list[float]] = {p.name: [] for p in probes.probes}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def record(*_):
        for p in probes.probes:
            norms[p.name].append(p.weight.grad.norm().item())

    optimizer.register_step_pre_hook(record)
    trainer = Trainer(model=model, loaders=loaders, task=evaluator.classification(),
                      optimizer=optimizer, scheduler=step_schedule(optimizer, milestones=[]),
                      max_steps=4, output_dir=tmp_path, device=CPU, log_every=4, probes=probes)
    trainer.fit()
    (row,) = [row for row in read_metrics(tmp_path) if row["split"] == "train"]
    assert len(norms) == 3
    for name, values in norms.items():
        assert row[f"{name}/conv1_weight_grad_norm"] == pytest.approx(values[-1], rel=1e-5)
        assert row[f"{name}/conv1_weight_grad_norm_mean"] == pytest.approx(sum(values) / 4, rel=1e-5)


def test_evaluation_rows_report_how_long_evaluation_took(tmp_path):
    loaders = {"train": DataLoader(toy_data(), batch_size=32),
               "val": DataLoader(toy_data(seed=1), batch_size=32)}
    trainer = toy_trainer(tmp_path, loaders, max_steps=2, eval_every=1)
    results = trainer.fit()
    val_rows = [row for row in read_metrics(tmp_path) if row["split"] == "val"]
    assert all(row["seconds"] >= 0 for row in val_rows)
    assert "seconds" not in results["val"]  # the returned metrics stay comparable


# ---------------------------------------------------------------------------
# The wandb logger
# ---------------------------------------------------------------------------


def test_rows_become_metrics_named_by_split():
    row = {"step": 300, "split": "val", "loss": 1.5, "error": 12.0}
    assert to_wandb(row) == {"step": 300, "val/loss": 1.5, "val/error": 12.0}


def test_runs_are_tagged_with_dataset_model_and_normalization_layer():
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=["+experiment=imagenet_resnet"])
    choices = {"model": "imagenet_resnet", "data": "imagenet", "experiment": "imagenet_resnet"}
    assert run_tags(cfg, choices) == ["imagenet", "imagenet_resnet", "BatchNorm2d"]


def test_config_has_wandb_settings():
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config")
    assert cfg.wandb.mode in {"online", "offline", "disabled"}
    assert cfg.wandb.project == "nstudy"


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "wandb-config"))
    return dict(project="nstudy-test", mode="offline")


def test_wandb_logger_logs_rows_and_summary_offline(tmp_path, offline):
    logger = WandbLogger(tmp_path, config={"seed": 3}, **offline)
    try:
        assert logger.run.name == tmp_path.name
        assert logger.run.config["seed"] == 3
        logger.log({"step": 5, "split": "val", "error": 9.5})
        logger.summary({"parameters": 123})
        assert logger.run.summary["val/error"] == 9.5
        assert logger.run.summary["parameters"] == 123
    finally:
        logger.finish()
    assert (tmp_path / "wandb").is_dir()  # offline files, for `wandb sync`


def test_wandb_logger_continues_the_run_of_its_directory(tmp_path, offline):
    """Resuming a run in the same directory continues the same wandb run."""
    first = WandbLogger(tmp_path, config={}, **offline)
    run_id = first.run.id
    first.finish()
    second = WandbLogger(tmp_path, config={}, **offline)
    try:
        assert second.run.id == run_id
    finally:
        second.finish()
    other = WandbLogger(tmp_path / "other", config={}, **offline)
    try:
        assert other.run.id != run_id
    finally:
        other.finish()


def test_wandb_logger_logs_probe_histograms_offline(tmp_path, offline):
    logger = WandbLogger(tmp_path, config={}, **offline)
    try:
        logger.log_histograms(10, {"stage1.block1.layer2/relu_input": {"counts": [1, 3, 2],
                                                           "edges": [0.0, 1.0, 2.0, 3.0]}})
    finally:
        logger.finish()


def test_transformer_sublayer_weights_get_their_training_gradient_norms(tmp_path):
    """The attention out_proj and feed-forward linear2 weights the probes watch."""
    from nstudy.core.model import Transformer
    from nstudy.core.probes import ProbeRecorder, model_probes

    torch.manual_seed(0)
    model = Transformer(30, d_model=16, num_heads=2, num_layers=3, d_ff=32)
    batches = [{"src": torch.randint(3, 30, (4, 7)), "tgt_in": torch.randint(3, 30, (4, 6)),
                "tgt_out": torch.randint(3, 30, (4, 6))} for _ in range(2)]
    probes = ProbeRecorder(model, model_probes(model), batch=batches[0],
                           histogram_every=100, stats_every=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    norms: dict[str, list[float]] = {}

    def record(*_):
        for p in probes.probes:
            norms.setdefault(f"{p.name}/{p.weight_name}", []).append(p.weight.grad.norm().item())

    optimizer.register_step_pre_hook(record)
    trainer = Trainer(model=model, loaders={"train": batches}, task=evaluator.translation(),
                      optimizer=optimizer, scheduler=step_schedule(optimizer, milestones=[]),
                      max_steps=2, output_dir=tmp_path, device=CPU, log_every=2, probes=probes)
    trainer.fit()
    (row,) = [row for row in read_metrics(tmp_path) if row["split"] == "train"]
    assert "decoder.layer3.norm2/cross_attn_out_proj_weight" in norms
    assert len(norms) == 15  # 3 encoder layers x 2 norms + 3 decoder layers x 3 norms
    for key, values in norms.items():
        assert row[f"{key}_grad_norm"] == pytest.approx(values[-1], rel=1e-5)
        assert row[f"{key}_grad_norm_mean"] == pytest.approx(sum(values) / 2, rel=1e-5)


def test_plain_nets_are_tagged():
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config",
                            overrides=["+experiment=cifar10_resnet", "model.residual=false"])
    choices = {"model": "cifar_resnet", "data": "cifar10"}
    assert run_tags(cfg, choices) == ["cifar10", "cifar_resnet", "plain", "BatchNorm2d"]
