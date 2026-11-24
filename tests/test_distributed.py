"""Tests for training on several GPUs with DistributedDataParallel.

Configs give batch sizes as global totals (the papers' numbers), and each of
the N processes handles 1/N of every batch. Training on N processes must then
reach the same weights as training on one, and evaluation must report the
same metrics.

The distributed tests start 2 CPU processes that talk over gloo, so they run
without GPUs.
"""

import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import TensorDataset

from nstudy.core import data, distributed, evaluator
from nstudy.core.probes import Probe, ProbeRecorder
from nstudy.core.trainer import Trainer, step_schedule

CPU = torch.device("cpu")
WORLD = 2


def toy_data(count: int, seed: int = 0) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(count, 8, generator=generator)
    return TensorDataset(inputs, (inputs[:, 0] > 0).long())


def toy_trainer(output_dir: Path, loaders: dict, max_steps: int, **kwargs) -> Trainer:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    return Trainer(
        model=model,
        loaders=loaders,
        task=evaluator.classification(),
        optimizer=optimizer,
        scheduler=step_schedule(optimizer, milestones=[3]),
        max_steps=max_steps,
        output_dir=output_dir,
        device=CPU,
        **kwargs,
    )


def read_metrics(directory: Path) -> list[dict]:
    return [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]


def spawn(function, tmp_path: Path, *args) -> list:
    """Run ``function(rank, tmp_path, *args)`` on WORLD processes; return what each returned."""
    mp.spawn(_entry, args=(function, str(tmp_path), args), nprocs=WORLD, join=True)
    return [torch.load(tmp_path / f"result-{rank}.pt") for rank in range(WORLD)]


def _entry(rank: int, function, tmp: str, args: tuple) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp}/store", rank=rank, world_size=WORLD
    )
    try:
        torch.save(function(rank, Path(tmp), *args), Path(tmp) / f"result-{rank}.pt")
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Samplers (no processes needed)
# ---------------------------------------------------------------------------


def test_shard_sampler_gives_every_index_to_exactly_one_rank():
    shards = [list(data.ShardSampler(11, rank, 3)) for rank in range(3)]
    assert sorted(index for shard in shards for index in shard) == list(range(11))
    assert [len(data.ShardSampler(11, rank, 3)) for rank in range(3)] == [4, 4, 3]


def test_token_batches_are_split_between_ranks_in_equal_numbers():
    """Every rank must make the same number of updates, or DDP waits forever."""
    lengths = [((i * 7) % 50) + 2 for i in range(1000)]
    whole = data.TokenBatchSampler(lengths, lengths, max_tokens=300, seed=3)
    ranks = [
        data.TokenBatchSampler(lengths, lengths, max_tokens=300, seed=3, rank=r, world_size=3)
        for r in range(3)
    ]
    for _ in range(2):  # two passes, each shuffled differently
        shards = [list(sampler) for sampler in ranks]
        assert len({len(shard) for shard in shards}) == 1
        assert len(shards[0]) == len(ranks[0]) == len(whole) // 3
        batches = [tuple(batch) for shard in shards for batch in shard]
        assert len(set(batches)) == len(batches)  # no batch on two ranks
        assert set(batches) <= {tuple(batch) for batch in whole.batches}


def test_one_process_is_not_distributed():
    assert not distributed.is_distributed()
    assert (distributed.rank(), distributed.world_size()) == (0, 1)
    assert distributed.is_main()
    assert distributed.per_rank(256) == 256


# ---------------------------------------------------------------------------
# Several processes
# ---------------------------------------------------------------------------


def _per_rank_batches(rank, tmp):
    loader = data.make_loader(toy_data(64), batch_size=32, train=True, num_workers=0)
    with pytest.raises(ValueError):
        distributed.per_rank(33)
    return {
        "batch": [len(labels) for _, labels in loader],
        "indices": list(loader.sampler),
        "gathered": distributed.gather_lists([f"rank{rank}"]),
    }


def test_each_rank_gets_its_share_of_the_global_batch(tmp_path):
    results = spawn(_per_rank_batches, tmp_path)
    assert all(result["batch"] == [16, 16] for result in results)
    first, second = (set(result["indices"]) for result in results)
    assert first.isdisjoint(second) and first | second == set(range(64))
    assert all(result["gathered"] == ["rank0", "rank1"] for result in results)


def _train(rank, tmp, max_steps, accumulate):
    loaders = {
        "train": data.make_loader(toy_data(64), batch_size=64 // accumulate, train=True,
                                  num_workers=0),
        "val": data.make_loader(toy_data(33, seed=1), batch_size=8, train=False, num_workers=0),
    }
    trainer = toy_trainer(tmp / "run", loaders, max_steps, accumulate=accumulate,
                          eval_every=2, log_every=1)
    results = trainer.fit()
    return {
        "results": results,
        "state": {key: value.clone() for key, value in trainer.module.state_dict().items()},
    }


@pytest.mark.parametrize("accumulate", [1, 2])
def test_two_processes_train_like_one(tmp_path, accumulate):
    """2 ranks x 32 images per update must reach the same weights as 1 process x 64.

    BatchNorm in train mode genuinely differs (each rank normalizes with the
    statistics of its own 32 images), so here it uses fixed statistics; the
    test then checks that gradients are averaged exactly across ranks.
    """
    results = spawn(_train_frozen_bn, tmp_path, 5, accumulate)
    single = _train_frozen_bn(0, tmp_path / "single", 5, 1)

    for result in results:
        for key, value in single["state"].items():
            assert torch.allclose(result["state"][key], value, atol=1e-5), key
        # Evaluation over sharded data matches evaluation in one process.
        for split, metrics in single["results"].items():
            assert result["results"][split] == pytest.approx(metrics)

    # Only the first rank writes files, so each line appears once.
    rows = read_metrics(tmp_path / "run")
    assert [row["step"] for row in rows if row["split"] == "val"] == [2, 4, 5]
    assert (tmp_path / "run" / "checkpoint.pt").exists()

    # The logged training loss and error cover the images of every rank.
    def train_rows(directory):
        return [value for row in read_metrics(directory) if row["split"] == "train"
                for value in (row["step"], row["loss"], row["error"])]

    assert train_rows(tmp_path / "run") == pytest.approx(train_rows(tmp_path / "single" / "run"))


def _train_frozen_bn(rank, tmp, max_steps, accumulate):
    """``_train`` with BatchNorm using fixed statistics, so ranks compute the same function."""
    original = nn.BatchNorm1d.forward

    def frozen_bn(self, x):
        self.eval()
        return original(self, x)

    nn.BatchNorm1d.forward = frozen_bn
    try:
        return _train(rank, tmp, max_steps, accumulate)
    finally:
        nn.BatchNorm1d.forward = original


def _evaluate(rank, tmp):
    torch.manual_seed(0)
    model = nn.Linear(8, 2)
    loader = data.make_loader(toy_data(65, seed=2), batch_size=8, train=False, num_workers=0)
    return evaluator.evaluate(model, loader, evaluator.classification(), CPU)


def test_evaluation_over_uneven_shards_matches_one_process(tmp_path):
    results = spawn(_evaluate, tmp_path)
    single = _evaluate(0, tmp_path)
    for result in results:
        assert result == pytest.approx(single)


def _batch_norm_statistics(rank, tmp):
    loaders = {"train": data.make_loader(toy_data(64), batch_size=64, train=True,
                                         num_workers=0)}
    trainer = toy_trainer(tmp / "bn", loaders, max_steps=3)
    trainer.fit()
    return {key: value for key, value in trainer.module.state_dict().items() if "running" in key}


def test_batch_norm_running_statistics_agree_across_ranks_after_training(tmp_path):
    """Each rank normalizes with its own 32 images, but the stored statistics are rank 0's."""
    first, second = spawn(_batch_norm_statistics, tmp_path)
    for key in first:
        assert torch.equal(first[key], second[key]), key


def _train_with_probes(rank, tmp, probes):
    loaders = {"train": data.make_loader(toy_data(64), batch_size=32, train=True,
                                         num_workers=0)}
    trainer = toy_trainer(tmp / ("probed" if probes else "plain"), loaders, max_steps=4,
                          log_every=2, eval_every=2)
    if probes:
        model = trainer.module
        probe = Probe("block", model[1], "relu_input", model[0].weight, "linear_weight")
        batch = (toy_data(16, seed=5).tensors[0], toy_data(16, seed=5).tensors[1])
        trainer.probes = ProbeRecorder(model, [probe], batch, histogram_every=2, stats_every=2)
    trainer.fit()
    return {key: value.clone() for key, value in trainer.module.state_dict().items()}


def test_probes_run_on_rank_0_without_disturbing_distributed_training(tmp_path):
    probed = spawn(_train_with_probes, tmp_path, True)
    plain = spawn(_train_with_probes, tmp_path, False)
    for ours, reference in zip(probed, plain):
        for key in reference:
            assert torch.equal(ours[key], reference[key]), key
    rows = read_metrics(tmp_path / "probed")
    assert [row["step"] for row in rows if row["split"] == "probe"] == [2, 4]
    assert len((tmp_path / "probed" / "histograms.jsonl").read_text().splitlines()) == 2
