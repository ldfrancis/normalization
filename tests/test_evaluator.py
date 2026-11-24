"""Tests for the tasks, the metrics and beam search in nstudy.core.evaluator."""

import itertools
import math

import pytest
import sacrebleu
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nstudy.core import data, evaluator
from nstudy.core.constants import BOS_ID, EOS_ID, PAD_ID
from nstudy.core.model import Transformer

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Classification: loss and error
# ---------------------------------------------------------------------------


def test_classification_reports_loss_and_error_in_percent():
    torch.manual_seed(0)
    logits, labels = torch.randn(50, 10), torch.randint(0, 10, (50,))
    loader = DataLoader(TensorDataset(logits, labels), batch_size=16)
    metrics = evaluator.evaluate(nn.Identity(), loader, evaluator.classification(), CPU)
    wrong = int((logits.argmax(1) != labels).sum())
    assert set(metrics) == {"loss", "error"}
    assert metrics["loss"] == pytest.approx(F.cross_entropy(logits, labels).item(), rel=1e-5)
    assert metrics["error"] == pytest.approx(100 * wrong / 50)


def test_classification_error_is_0_when_all_right_and_100_when_all_wrong():
    labels = torch.arange(10)
    right = F.one_hot(labels, 10).float()
    wrong = F.one_hot((labels + 1) % 10, 10).float()
    for logits, expected in ((right, 0.0), (wrong, 100.0)):
        loader = DataLoader(TensorDataset(logits, labels), batch_size=4)
        metrics = evaluator.evaluate(nn.Identity(), loader, evaluator.classification(), CPU)
        assert metrics["error"] == expected


def test_evaluate_restores_training_mode_and_does_not_smooth():
    logits, labels = torch.randn(8, 4), torch.randint(0, 4, (8,))
    loader = DataLoader(TensorDataset(logits, labels), batch_size=8)
    model = nn.Identity().train()
    smoothed = evaluator.classification(label_smoothing=0.1)
    metrics = evaluator.evaluate(model, loader, smoothed, CPU)
    assert model.training
    assert metrics["loss"] == pytest.approx(F.cross_entropy(logits, labels).item(), rel=1e-5)


# ---------------------------------------------------------------------------
# Translation: loss
# ---------------------------------------------------------------------------


class FixedLogits(nn.Module):
    """A translation 'model' that returns prepared logits for each call."""

    def __init__(self, outputs: list[torch.Tensor]) -> None:
        super().__init__()
        self.outputs = list(outputs)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        return self.outputs.pop(0)


def test_translation_loss_averages_over_non_padding_tokens():
    torch.manual_seed(0)
    # Two batches with different shapes and different amounts of padding.
    targets = [torch.tensor([[5, 6, 7], [8, 9, PAD_ID]]), torch.tensor([[4, PAD_ID, PAD_ID, PAD_ID]])]
    logits = [torch.randn(*target.shape, 12) for target in targets]
    batches = [{"src": target, "tgt_in": target, "tgt_out": target} for target in targets]
    metrics = evaluator.evaluate(FixedLogits(logits), batches, evaluator.translation(), CPU)

    flat_logits = torch.cat([x.flatten(0, 1) for x in logits])
    flat_targets = torch.cat([t.flatten() for t in targets])
    real = flat_targets != PAD_ID
    assert int(real.sum()) == 6
    expected = F.cross_entropy(flat_logits[real], flat_targets[real]).item()
    # Without a tokenizer there is no text to score, so only the loss is reported.
    assert set(metrics) == {"loss"}
    assert metrics["loss"] == pytest.approx(expected, rel=1e-5)


def test_translation_training_loss_ignores_padding_and_smooths_labels():
    torch.manual_seed(0)
    targets = torch.tensor([5, 6, PAD_ID, PAD_ID])
    logits = torch.randn(4, 12)
    task = evaluator.translation(label_smoothing=0.1)
    loss = task.training_loss(logits, targets)

    changed = logits.clone()
    changed[2:] = torch.randn(2, 12)
    assert task.training_loss(changed, targets) == pytest.approx(loss.item())
    expected = F.cross_entropy(logits[:2], targets[:2], label_smoothing=0.1)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)
    assert loss.item() != pytest.approx(F.cross_entropy(logits[:2], targets[:2]).item())


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

WORDS = [3, 4]  # the only tokens the Markov model can produce besides EOS


class MarkovModel(nn.Module):
    """The next token's distribution depends only on the previous token."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        table = torch.randn(5, 5, generator=torch.Generator().manual_seed(seed)) * 2
        table[:, [PAD_ID, BOS_ID]] = -1e9  # never produced
        self.table = table

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        return torch.zeros(src.shape[0], 1, 1)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        return self.table[tgt]

    def best_by_enumeration(self, max_len: int, alpha: float) -> list[int]:
        """Score every sentence of up to ``max_len`` tokens and return the best."""
        log_probs = self.table.log_softmax(-1)
        best, best_score = None, -math.inf
        for words in range(max_len):
            for prefix in itertools.product(WORDS, repeat=words):
                sentence = [*prefix, EOS_ID]
                path = [BOS_ID, *sentence]
                total = sum(log_probs[a, b].item() for a, b in zip(path, path[1:]))
                score = total / evaluator.length_normalizer(len(sentence), alpha)
                if score > best_score:
                    best, best_score = sentence, score
        return best


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("alpha", [0.0, 0.6, 3.0])
def test_wide_beam_finds_the_best_scoring_sentence(seed, alpha):
    """With a beam wide enough to keep everything, the search is exhaustive."""
    model = MarkovModel(seed)
    src = torch.full((1, 1), 3)
    out = evaluator.beam_search(model, src, beam_size=8, length_penalty=alpha, max_extra_length=2)
    sentence = out[0].tolist()
    sentence = sentence[: sentence.index(EOS_ID) + 1]
    assert sentence == model.best_by_enumeration(max_len=3, alpha=alpha)


def test_length_penalty_changes_which_sentence_wins():
    """A larger alpha favours longer sentences."""
    lengths = {alpha: [] for alpha in (0.0, 3.0)}
    for seed in range(8):
        model = MarkovModel(seed)
        for alpha in lengths:
            lengths[alpha].append(len(model.best_by_enumeration(max_len=3, alpha=alpha)))
    assert sum(lengths[3.0]) > sum(lengths[0.0])
    assert evaluator.length_normalizer(1, 0.6) == pytest.approx(1.0)
    assert evaluator.length_normalizer(7, 0.6) == pytest.approx(2.0**0.6)


class CopyModel(nn.Module):
    """A perfect 'translator' whose output is its input."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        return src

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tgt.shape[1]).clamp(max=src.shape[1] - 1)
        return F.one_hot(src[:, positions], self.vocab_size).float() * 30

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        return self.decode(tgt_in, self.encode(src), src)


def test_beam_search_handles_sentences_of_different_lengths_in_one_batch():
    src = torch.tensor(
        [[5, 6, 7, 8, EOS_ID], [9, EOS_ID, PAD_ID, PAD_ID, PAD_ID], [4, 4, 4, EOS_ID, PAD_ID]]
    )
    out = evaluator.beam_search(CopyModel(12), src, beam_size=4)
    assert out.shape[0] == 3
    # Every row ends with EOS and is padded afterwards; no BOS is returned.
    assert torch.equal(out, src[:, : out.shape[1]])
    assert BOS_ID not in out


def test_beam_search_stops_at_source_length_plus_max_extra_length():
    """A model that keeps repeating one token is cut off at the length limit."""

    class Repeats(CopyModel):
        def __init__(self, vocab_size: int, end_logit: float) -> None:
            super().__init__(vocab_size)
            self.end_logit = end_logit

        def decode(self, tgt, memory, src):
            logits = F.one_hot(torch.full_like(tgt, 7), self.vocab_size).float() * 30
            logits[..., EOS_ID] = self.end_logit
            return logits

    src = torch.tensor([[5, 6, EOS_ID]])
    limit = 3 + 4
    # Ending is very unlikely: the sentence is ended only at the last allowed position.
    out = evaluator.beam_search(Repeats(12, 0.0), src, beam_size=2, max_extra_length=4)
    assert out.tolist() == [[7] * (limit - 1) + [EOS_ID]]
    # Ending is impossible: the unfinished sentence is returned at full length.
    out = evaluator.beam_search(Repeats(12, -torch.inf), src, beam_size=2, max_extra_length=4)
    assert out.tolist() == [[7] * limit]


def test_beam_search_runs_on_the_transformer():
    torch.manual_seed(0)
    model = Transformer(30, d_model=32, num_heads=2, num_layers=1, d_ff=64).eval()
    src = torch.randint(3, 30, (4, 6))
    out = evaluator.beam_search(model, src, beam_size=4, max_extra_length=5)
    assert out.shape[0] == 4
    assert 1 <= out.shape[1] <= 6 + 5
    assert out.dtype == torch.long
    for row in out.tolist():
        if EOS_ID in row:
            assert set(row[row.index(EOS_ID) + 1 :]) <= {PAD_ID}


# ---------------------------------------------------------------------------
# Translation: BLEU
# ---------------------------------------------------------------------------

SENTENCES = [
    "the small dog sleeps in my green garden",
    "your cat runs on the house",
    "a dog was in the garden and the cat was on the house",
    "my house is small",
    "the green garden is in your house, is the dog in my garden?",
    "the cat sleeps and the dog runs in the small garden",
]


@pytest.fixture(scope="module")
def tokenizer_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("tokenizer") / "bpe.json"
    data.train_bpe_tokenizer(SENTENCES, vocab_size=300).save(str(path))
    return path


def make_batches(tokenizer_file, sources: list[str], targets: list[str]) -> list[dict]:
    tokenizer = evaluator.Translator(tokenizer_file).tokenizer
    rows = [
        {"src": tokenizer.encode(s).ids + [EOS_ID], "tgt": tokenizer.encode(t).ids + [EOS_ID]}
        for s, t in zip(sources, targets)
    ]
    return [data.collate_translation(rows[:4]), data.collate_translation(rows[4:])]


def test_translator_turns_token_ids_back_into_sentences(tokenizer_file):
    translator = evaluator.Translator(tokenizer_file)
    batch = make_batches(tokenizer_file, SENTENCES, SENTENCES)[0]
    assert translator.to_text(batch["tgt_out"]) == SENTENCES[:4]
    assert translator.translate(CopyModel(300), batch["src"]) == SENTENCES[:4]


def test_perfect_translations_score_100_bleu(tokenizer_file):
    task = evaluator.translation(tokenizer_file=tokenizer_file)
    batches = make_batches(tokenizer_file, SENTENCES, SENTENCES)
    metrics = evaluator.evaluate(CopyModel(300), batches, task, CPU)
    assert set(metrics) == {"loss", "bleu"}
    assert metrics["bleu"] == pytest.approx(100.0)
    assert metrics["loss"] < 1e-6


def test_bleu_is_the_sacrebleu_corpus_score_of_the_translations(tokenizer_file):
    """The copy model 'translates' each source into itself; score that against other references."""
    references = [
        "the small dog sleeps in your green garden",
        "your cat runs on the house",
        "a cat was in the garden and the dog was on the house",
        "my garden is small",
        "the green garden is in your house, is the cat in my garden?",
        "the cat runs and the dog sleeps in the small garden",
    ]
    task = evaluator.translation(tokenizer_file=tokenizer_file)
    batches = make_batches(tokenizer_file, SENTENCES, references)
    metrics = evaluator.evaluate(CopyModel(300), batches, task, CPU)
    expected = sacrebleu.corpus_bleu(SENTENCES, [references]).score
    assert 0 < expected < 100
    assert metrics["bleu"] == pytest.approx(expected)


def test_translation_task_carries_the_beam_search_settings(tokenizer_file):
    task = evaluator.translation(
        label_smoothing=0.1, tokenizer_file=tokenizer_file, beam_size=5, length_penalty=1.0
    )
    assert task.ignore_index == PAD_ID
    assert not task.report_error
    assert (task.translator.beam_size, task.translator.length_penalty) == (5, 1.0)
    assert evaluator.classification().translator is None
    assert evaluator.classification().report_error
