"""Record what normalization layers of a trained model do, before their scale and shift.

    python nstudy/norm_io.py run_dir=outputs/baselines/cifar10_resnet56_seed0
    python nstudy/norm_io.py run_dir=... split=test examples=1000 mode=train

Loads the model of a finished training run (``<run_dir>/.hydra/config.yaml``
and ``checkpoint.pt``), passes ``examples`` held-out examples through it and,
for the norms of an early, a middle and a late block, records the input ``x``,
the normalized ``x_hat`` (before scale and shift) and the output ``y`` of many
elements (see ``nstudy/core/norm_io.py``). Writes to
``<run_dir>/norm_io/<split>_<mode>/``:

* ``points.pt``: the points of every layer (``torch.load`` it to make other plots);
* ``norm_io.png``: ``x_hat`` against ``x`` for every layer;
* ``summary.json``: the layers, their positions and the number of examples.

Settings are in ``config/norm_io.yaml``.
"""

import json
import logging
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from nstudy.core import norm_io
from nstudy.core.probes import model_probes
from nstudy.core.trainer import pick_device

log = logging.getLogger(__name__)


def load_run(run_dir: Path, checkpoint: str | None, batch_size: int | None, device: torch.device):
    """The trained model, the data loaders and the task of a training run."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    if batch_size is not None:
        cfg.data.batch_size = batch_size
    torch.manual_seed(cfg.seed)
    model = instantiate(cfg.model)
    path = Path(checkpoint) if checkpoint else run_dir / "checkpoint.pt"
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device)
    loaders = instantiate(cfg.data)  # before the task: it may create the tokenizer
    loaders.pop("probe", None)
    task = instantiate(cfg.trainer.task)
    return cfg, model, loaders, task, state["step"]


@hydra.main(config_path="../config", config_name="norm_io", version_base=None)
def main(cfg: DictConfig) -> None:
    run_dir = Path(cfg.run_dir)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    device = pick_device()
    run_cfg, model, loaders, task, step = load_run(run_dir, cfg.checkpoint, cfg.batch_size, device)
    if cfg.split not in loaders or cfg.split == "train":
        held_out = sorted(split for split in loaders if split != "train")
        raise ValueError(f"split must be a held-out split of this run: {held_out}")
    loader = loaders[cfg.split]

    blocks = None if cfg.blocks is None else list(cfg.blocks)
    probes = model_probes(model, blocks)
    first = next(iter(loader))
    quota = norm_io.per_batch_quota(cfg.max_points, cfg.examples, norm_io.batch_size(first))
    recorder = norm_io.NormIORecorder(model, probes, quota, mode=cfg.mode, seed=cfg.seed)
    examples = recorder.run(loader, task, device, cfg.examples)
    recorder.remove()
    log.info("recorded %d layers on %d %s examples (mode %s)", len(probes), examples, cfg.split, cfg.mode)

    metadata = {
        "run_dir": str(run_dir),
        "step": step,
        "split": cfg.split,
        "mode": cfg.mode,
        "examples": examples,
        "model": run_cfg.model._target_.rsplit(".", 1)[-1],
        "norm_layer": type(probes[0].module).__name__,
    }
    norm_io.save(recorder.records, output_dir / "points.pt", metadata)
    layers = {
        name: {"norm": record.norm, "position": record.position, **record.tensors()}
        for name, record in recorder.records.items()
    }
    title = (f"{run_dir.name}: {metadata['norm_layer']} before scale and shift "
             f"({cfg.split}, {examples} examples, {cfg.mode} mode, step {step})")
    norm_io.plot(layers, output_dir / "norm_io.png", channels=cfg.channels, title=title)
    summary = {
        **metadata,
        "layers": {name: {"position": layer["position"], "norm": layer["norm"],
                          "points": int(layer["x"].numel())} for name, layer in layers.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("wrote %s", output_dir)

    if cfg.wandb.mode != "disabled":
        from nstudy.core.tracking import WandbLogger

        logger = WandbLogger(
            output_dir,
            config={**OmegaConf.to_container(cfg, resolve=True), **metadata},
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=run_cfg.get("wandb", {}).get("group"),
            name=f"{run_dir.name}-norm_io-{cfg.split}-{cfg.mode}",
            tags=["norm_io", metadata["model"], metadata["norm_layer"]],
            mode=cfg.wandb.mode,
            job_type="norm_io",
        )
        try:
            logger.log_images({"norm_io": output_dir / "norm_io.png"})
            if cfg.wandb.log_points:
                logger.log_file(output_dir / "points.pt", name=f"{logger.run.name}-points",
                                kind="norm_io")
        finally:
            logger.finish()


if __name__ == "__main__":
    main()
