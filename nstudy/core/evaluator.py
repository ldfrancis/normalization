"""How a batch is scored, and evaluation of a model over a dataloader.

A ``Task`` turns a batch into logits and targets, so classification and
translation share one training loss (cross-entropy). Evaluation reports the
loss plus the metric of each paper: classification error for the ResNets and
BLEU for the Transformer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import sacrebleu
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import Tensor, nn
from torch.utils.data import DataLoader

from nstudy.core import distributed
from nstudy.core.constants import BOS_ID, EOS_ID, PAD_ID

Forward = Callable[[nn.Module, Any, torch.device], tuple[Tensor, Tensor]]


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


def length_normalizer(length: int, alpha: float) -> float:
    """Length penalty of Wu et al. (2016): ``((5 + length) / 6) ** alpha``."""
    return ((5.0 + length) / 6.0) ** alpha


@torch.no_grad()
def beam_search(
    model: nn.Module,
    src: Tensor,
    beam_size: int = 4,
    length_penalty: float = 0.6,
    max_extra_length: int = 50,
) -> Tensor:
    """Translate source token ids (B, S) and return the best hypothesis of each row.

    Hypotheses are ranked by ``log-probability / length_normalizer(length)``,
    where the length counts the end-of-sentence token. A hypothesis has at most
    ``S + max_extra_length`` tokens. The search for a batch stops as soon as no
    unfinished hypothesis can still beat the best finished one.

    Returns token ids (B, L) without the beginning-of-sentence token. Each row
    ends with the end-of-sentence token followed by padding, unless it reached
    the maximum length unfinished.

    ``model`` needs ``encode(src)`` and ``decode(tgt, memory, src)``, as
    ``nstudy.core.model.Transformer`` has.
    """
    batch, src_len = src.shape
    device = src.device
    max_len = src_len + max_extra_length
    memory = model.encode(src).repeat_interleave(beam_size, dim=0)
    src = src.repeat_interleave(beam_size, dim=0)
    rows = torch.arange(batch, device=device)[:, None]

    # Unfinished hypotheses, ranked by total log-probability. Only the first
    # beam starts alive, so the first step does not pick the same token K times.
    alive = torch.full((batch, beam_size, 1), BOS_ID, dtype=torch.long, device=device)
    alive_scores = torch.full((batch, beam_size), -torch.inf, device=device)
    alive_scores[:, 0] = 0.0
    # Finished hypotheses, ranked by length-normalized log-probability.
    finished = torch.full((batch, beam_size, 1), PAD_ID, dtype=torch.long, device=device)
    finished_scores = torch.full((batch, beam_size), -torch.inf, device=device)

    for length in range(1, max_len + 1):
        logits = model.decode(alive.flatten(0, 1), memory, src)[:, -1]
        log_probs = logits.log_softmax(dim=-1).view(batch, beam_size, -1)
        vocab_size = log_probs.shape[-1]

        # The best 2K extensions always contain at least K that do not end,
        # because each hypothesis has only one extension that ends.
        candidates = (alive_scores[:, :, None] + log_probs).view(batch, -1)
        scores, flat = candidates.topk(2 * beam_size, dim=-1)
        beams, tokens = flat // vocab_size, flat % vocab_size
        sequences = torch.cat([alive[rows, beams], tokens[:, :, None]], dim=-1)
        ended = tokens == EOS_ID

        normalized = (scores / length_normalizer(length, length_penalty)).masked_fill(
            ~ended, -torch.inf
        )
        pool = torch.cat([F.pad(finished, (0, 1), value=PAD_ID), sequences], dim=1)
        pool_scores = torch.cat([finished_scores, normalized], dim=1)
        finished_scores, keep = pool_scores.topk(beam_size, dim=-1)
        finished = pool[rows, keep]

        alive_scores, keep = scores.masked_fill(ended, -torch.inf).topk(beam_size, dim=-1)
        alive = sequences[rows, keep]

        # Log-probabilities only fall as a hypothesis grows, so the most an
        # unfinished hypothesis can reach is its current score at full length.
        best_reachable = alive_scores[:, 0] / length_normalizer(max_len, length_penalty)
        if (finished_scores[:, 0] >= best_reachable).all():
            break

    nothing_finished = finished_scores[:, 0].isinf()
    best = torch.where(nothing_finished[:, None], alive[:, 0], finished[:, 0])
    return best[:, 1:]


class Translator:
    """Translate batches with beam search and turn token ids back into text.

    Args:
        tokenizer_file: tokenizer saved by ``nstudy.core.data.translation_loaders``.
            It is read on first use, so the file may be created later.
        beam_size, length_penalty, max_extra_length: see ``beam_search``.
    """

    def __init__(
        self,
        tokenizer_file: str | Path,
        beam_size: int = 4,
        length_penalty: float = 0.6,
        max_extra_length: int = 50,
    ) -> None:
        self.tokenizer_file = str(tokenizer_file)
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.max_extra_length = max_extra_length
        self._tokenizer: Tokenizer | None = None

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.from_file(self.tokenizer_file)
        return self._tokenizer

    def to_text(self, token_ids: Tensor) -> list[str]:
        """Decode rows of token ids, stopping each at its end-of-sentence token."""
        sentences = []
        for row in token_ids.tolist():
            sentences.append(row[: row.index(EOS_ID)] if EOS_ID in row else row)
        return self.tokenizer.decode_batch(sentences, skip_special_tokens=True)

    def translate(self, model: nn.Module, src: Tensor) -> list[str]:
        """Translate source token ids (B, S) into sentences."""
        return self.to_text(
            beam_search(model, src, self.beam_size, self.length_penalty, self.max_extra_length)
        )


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def classification_forward(
    model: nn.Module, batch: Any, device: torch.device
) -> tuple[Tensor, Tensor]:
    """``batch`` is ``(images, labels)``. Returns logits (B, C) and labels (B,)."""
    images, labels = batch
    return model(images.to(device)), labels.to(device)


def translation_forward(
    model: nn.Module, batch: Any, device: torch.device
) -> tuple[Tensor, Tensor]:
    """``batch`` has ``src``, ``tgt_in`` and ``tgt_out``. Returns one row per target token."""
    logits = model(batch["src"].to(device), batch["tgt_in"].to(device))
    return logits.flatten(0, 1), batch["tgt_out"].to(device).flatten()


@dataclass(frozen=True)
class Task:
    """What the model is trained to do and how it is evaluated.

    Args:
        forward: maps ``(model, batch, device)`` to logits (N, C) and targets (N,).
        ignore_index: target value that is excluded from the loss.
        label_smoothing: smoothing of the training loss. Evaluation never smooths.
        report_error: evaluation reports the classification error.
        translator: when set, evaluation reports the BLEU score of its translations.
    """

    forward: Forward
    ignore_index: int = -100
    label_smoothing: float = 0.0
    report_error: bool = False
    translator: Translator | None = None

    def training_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Mean cross-entropy over the targets that are not ignored."""
        return F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


def classification(label_smoothing: float = 0.0) -> Task:
    """Image classification, evaluated by the classification error."""
    return Task(classification_forward, label_smoothing=label_smoothing, report_error=True)


def translation(
    label_smoothing: float = 0.0,
    tokenizer_file: str | Path | None = None,
    beam_size: int = 4,
    length_penalty: float = 0.6,
    max_extra_length: int = 50,
) -> Task:
    """Translation, evaluated by BLEU when ``tokenizer_file`` is given.

    Without a tokenizer, token ids cannot be turned into text, so evaluation
    reports only the loss.
    """
    translator = None
    if tokenizer_file is not None:
        translator = Translator(tokenizer_file, beam_size, length_penalty, max_extra_length)
    return Task(
        translation_forward,
        ignore_index=PAD_ID,
        label_smoothing=label_smoothing,
        translator=translator,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, task: Task, device: torch.device
) -> dict[str, float]:
    """Evaluate ``model`` over ``loader``. The model's train/eval mode is restored.

    With several processes, each passes its own shard of the split (and the
    plain model, not the DDP wrapper); the metrics cover the whole split and
    every process gets them.

    Returns:
        ``loss``: mean cross-entropy per prediction (per image, or per
        non-padding target token), without label smoothing.

        ``error``: for classification, the percentage of images whose most
        likely class is wrong (top-1 error).

        ``bleu``: for translation, the corpus BLEU score (0 to 100) of the
        beam-search translations against the references, computed by sacreBLEU
        on detokenized text.
    """
    was_training = model.training
    model.eval()
    loss_sum, wrong, count = 0.0, 0, 0
    hypotheses: list[str] = []
    references: list[str] = []
    for batch in loader:
        logits, targets = task.forward(model, batch, device)
        counted = targets != task.ignore_index
        loss_sum += F.cross_entropy(
            logits, targets, ignore_index=task.ignore_index, reduction="sum"
        ).item()
        count += int(counted.sum())
        if task.report_error:
            wrong += int(((logits.argmax(dim=-1) != targets) & counted).sum())
        if task.translator is not None:
            hypotheses += task.translator.translate(model, batch["src"].to(device))
            references += task.translator.to_text(batch["tgt_out"])
    model.train(was_training)

    # With several processes, each scored its own shard of the split.
    loss_sum, wrong, count = distributed.all_reduce_sum([loss_sum, wrong, count], device)
    if task.translator is not None:
        hypotheses = distributed.gather_lists(hypotheses)
        references = distributed.gather_lists(references)

    metrics = {"loss": loss_sum / count}
    if task.report_error:
        metrics["error"] = 100.0 * wrong / count
    if task.translator is not None:
        metrics["bleu"] = sacrebleu.corpus_bleu(hypotheses, [references]).score
    return metrics
