"""Experiment tracking with Weights & Biases.

The trainer hands every row it writes to ``metrics.jsonl`` to a logger, which
here sends it to wandb as ``<split>/<metric>`` against the optimizer step.
Only rank 0 creates a logger. Set ``wandb.mode`` in the config:

* ``online``: stream to wandb (run ``wandb login`` once on the machine).
* ``offline``: write to ``<run dir>/wandb``; upload later with ``wandb sync``.
* ``disabled``: no wandb at all; ``metrics.jsonl`` is still written.
"""

from __future__ import annotations

import secrets
import string
from pathlib import Path
from typing import Any, Mapping, Sequence

RUN_ID_FILE = "wandb_run_id.txt"


def new_run_id() -> str:
    """A random wandb run id: 8 lowercase letters and digits."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(8))


def to_wandb(row: Mapping[str, Any]) -> dict[str, Any]:
    """``{"step": 3, "split": "val", "error": 9.5}`` -> ``{"step": 3, "val/error": 9.5}``."""
    split = row["split"]
    metrics = {f"{split}/{key}": value for key, value in row.items() if key not in ("step", "split")}
    return {"step": row["step"], **metrics}


def run_tags(cfg: Any, choices: Mapping[str, Any]) -> list[str]:
    """Tags to filter runs by: the dataset, the model config, ``plain`` for nets without
    shortcuts, and the normalization layer."""
    tags = [str(choices[group]) for group in ("data", "model") if choices.get(group)]
    if cfg.model.get("residual") is False:
        tags.append("plain")
    norm_layer = cfg.model.get("norm_layer")
    if norm_layer is not None and "_target_" in norm_layer:
        tags.append(str(norm_layer["_target_"]).rsplit(".", 1)[-1])
    return tags


class WandbLogger:
    """A wandb run tied to a run directory.

    The run's id is saved in the directory, so restarting in the same
    directory (to resume from its checkpoint) continues the same wandb run.

    Args:
        output_dir: the run directory; also where wandb keeps its local files.
        config: settings shown with the run (the resolved Hydra config).
        project, entity, group, notes: as in ``wandb.init``.
        name: run name; the run directory's name when None.
        tags: labels to filter runs by.
        mode: ``"online"``, ``"offline"`` or ``"disabled"``.
        job_type: what the run does (None for training; ``"norm_io"`` for the
            analysis of ``nstudy/norm_io.py``), to filter runs by.
    """

    def __init__(
        self,
        output_dir: str | Path,
        config: Mapping[str, Any],
        project: str = "nstudy",
        entity: str | None = None,
        group: str | None = None,
        name: str | None = None,
        tags: Sequence[str] = (),
        notes: str | None = None,
        mode: str = "online",
        job_type: str | None = None,
    ) -> None:
        import wandb

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        id_file = output_dir / RUN_ID_FILE
        run_id = id_file.read_text().strip() if id_file.exists() else new_run_id()
        self.run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            name=name or output_dir.name,
            tags=list(tags),
            notes=notes,
            config=dict(config),
            dir=str(output_dir),
            id=run_id,
            resume="allow",
            mode=mode,
            job_type=job_type,
        )
        id_file.write_text(run_id)
        # Plot everything against the optimizer step. A custom step metric
        # (rather than wandb's own step counter) lets a resumed run log steps
        # it had already logged before it stopped.
        self.run.define_metric("step")
        self.run.define_metric("*", step_metric="step")

    def log(self, row: Mapping[str, Any]) -> None:
        self.run.log(to_wandb(row))

    def log_histograms(self, step: int, histograms: Mapping[str, Mapping[str, list]]) -> None:
        """Histograms from ``nstudy.core.probes``, as ``histogram/<probe>/<tensor>``."""
        import numpy as np
        import wandb

        data: dict[str, Any] = {"step": step}
        for name, values in histograms.items():
            data[f"histogram/{name}"] = wandb.Histogram(
                np_histogram=(np.asarray(values["counts"]), np.asarray(values["edges"]))
            )
        self.run.log(data)

    def log_images(self, images: Mapping[str, str | Path]) -> None:
        """Image files (such as figures), each under its key."""
        import wandb

        self.run.log({key: wandb.Image(str(path)) for key, path in images.items()})

    def summary(self, values: Mapping[str, Any]) -> None:
        """Values shown in the runs table, such as the parameter count."""
        self.run.summary.update(dict(values))

    def log_file(self, path: str | Path, name: str, kind: str) -> None:
        """Upload a file (such as the final checkpoint) as a wandb artifact."""
        import wandb

        artifact = wandb.Artifact(name, type=kind)
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self, exit_code: int = 0) -> None:
        self.run.finish(exit_code=exit_code)
