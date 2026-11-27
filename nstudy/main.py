"""Train a model. Run from the project root.

One GPU (CIFAR):

    python nstudy/main.py +experiment=cifar10_resnet
    python nstudy/main.py +experiment=cifar10_resnet model.depth=110 trainer.scheduler.warmup_steps=400

Several GPUs (ImageNet, WMT14), one process per GPU:

    torchrun --standalone --nproc_per_node=8 nstudy/main.py +experiment=imagenet_resnet \\
        hydra.run.dir=outputs/imagenet_resnet50

Batch sizes in the configs are global, so the papers' settings hold for any
number of GPUs. Give ``hydra.run.dir`` explicitly with torchrun: every process
starts Hydra, and a shared directory keeps their files in one place.
``scripts/baselines.sh`` does all of this for the baseline runs.

Metrics go to ``metrics.jsonl`` in the run directory and to Weights & Biases
(``wandb.mode=online|offline|disabled``; see ``config/config.yaml``).
"""

import logging
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from nstudy.core import distributed
from nstudy.core.model import ResNet
from nstudy.core.probes import ProbeRecorder, model_probes
from nstudy.core.tracking import WandbLogger, run_tags
from nstudy.core.trainer import Logger, Trainer, layerwise_lr_groups

log = logging.getLogger(__name__)


def build_probes(cfg: DictConfig, model: torch.nn.Module, loaders: dict) -> ProbeRecorder | None:
    """Probes measured on the data's fixed ``"probe"`` batch of training examples."""
    probe_loader = loaders.pop("probe", None)  # not a held-out split
    if not cfg.probe.enabled or probe_loader is None:
        return None
    blocks = None if cfg.probe.blocks is None else list(cfg.probe.blocks)
    return ProbeRecorder(
        model,
        model_probes(model, blocks),
        batch=next(iter(probe_loader)),
        histogram_every=cfg.probe.histogram_every,
        stats_every=cfg.probe.stats_every,
        bins=cfg.probe.bins,
        seed=cfg.seed,
    )


def optimizer_params(cfg: DictConfig, model: torch.nn.Module) -> Any:
    """``model``'s parameters, in groups with a rate per layer if ``trainer.first_layer_lr_multiplier`` is set."""
    multiplier = cfg.trainer.get("first_layer_lr_multiplier")
    if multiplier is None:
        return model.parameters()
    if not isinstance(model, ResNet):
        raise ValueError("trainer.first_layer_lr_multiplier is only supported for ResNets")
    return layerwise_lr_groups(model, cfg.trainer.optimizer.lr, multiplier)


def build_trainer(cfg: DictConfig, output_dir: str, logger: Logger | None = None) -> Trainer:
    torch.manual_seed(cfg.seed)
    model = instantiate(cfg.model)
    # Data augmentation draws from this seed; a different one per process
    # keeps the processes from applying the same random crops.
    torch.manual_seed(cfg.seed + distributed.rank())
    # Built before the task: the translation loaders create the tokenizer it reads.
    # Rank 0 downloads and caches the data before the other processes read it.
    with distributed.main_process_first():
        loaders = instantiate(cfg.data)
    probes = build_probes(cfg, model, loaders)
    optimizer = instantiate(cfg.trainer.optimizer)(optimizer_params(cfg, model))
    scheduler = instantiate(cfg.trainer.scheduler)(optimizer)
    trainer = Trainer(
        model=model,
        loaders=loaders,
        task=instantiate(cfg.trainer.task),
        optimizer=optimizer,
        scheduler=scheduler,
        max_steps=cfg.trainer.max_steps,
        output_dir=output_dir,
        accumulate=cfg.trainer.accumulate,
        eval_every=cfg.trainer.eval_every,
        log_every=cfg.trainer.log_every,
        eval_splits=list(cfg.trainer.eval_splits),
        precision=cfg.trainer.precision,
        logger=logger,
        probes=probes,
    )
    if cfg.resume_from is not None:
        trainer.load_checkpoint(cfg.resume_from)
        log.info("resumed from %s at step %d", cfg.resume_from, trainer.step)
    return trainer


def start_wandb(cfg: DictConfig, output_dir: str) -> WandbLogger | None:
    """The wandb run of this training run, on rank 0 only, unless wandb is disabled."""
    if not distributed.is_main() or cfg.wandb.mode == "disabled":
        return None
    choices = dict(HydraConfig.get().runtime.choices)
    config = OmegaConf.to_container(cfg, resolve=True)
    config["world_size"] = distributed.world_size()
    config["choices"] = choices
    return WandbLogger(
        output_dir,
        config=config,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        name=cfg.wandb.name,
        tags=run_tags(cfg, choices) + list(cfg.wandb.tags),
        notes=cfg.wandb.notes,
        mode=cfg.wandb.mode,
    )


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    distributed.setup()
    logger = None
    try:
        if not distributed.is_main():
            logging.getLogger().setLevel(logging.WARNING)
        output_dir = distributed.broadcast_object(HydraConfig.get().runtime.output_dir)
        logger = start_wandb(cfg, output_dir)
        trainer = build_trainer(cfg, output_dir, logger)
        parameters = sum(p.numel() for p in trainer.module.parameters())
        log.info(
            "training %d parameters on %d process(es), device %s, output %s",
            parameters, distributed.world_size(), trainer.device, output_dir,
        )
        if logger is not None:
            logger.summary({"parameters": parameters, "device": str(trainer.device)})
        trainer.fit()
        if logger is not None and cfg.wandb.log_checkpoint:
            logger.log_file(trainer.checkpoint_path, name=f"{logger.run.name}-model", kind="model")
    except BaseException:
        if logger is not None:
            logger.finish(exit_code=1)
            logger = None
        raise
    finally:
        if logger is not None:
            logger.finish()
        distributed.cleanup()


if __name__ == "__main__":
    main()
