"""Tests for the datasets and dataloaders in nstudy.core.data.

Nothing here downloads data. The CIFAR tests that need the real files are
skipped until the datasets exist under ``data/``.
"""

import inspect
import random
from pathlib import Path

import datasets as hf_datasets
import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer

from nstudy.core import data
from nstudy.core.constants import (
    BOS_ID,
    CIFAR10_MEAN,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
)
from nstudy.core.model import Transformer

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "config"
DATA_DIR = PROJECT_DIR / "data"


def random_image(size: int) -> Image.Image:
    pixels = np.random.default_rng(0).integers(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


# ---------------------------------------------------------------------------
# CIFAR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["cifar10", "cifar100"])
@pytest.mark.parametrize("train", [True, False])
def test_cifar_transform_returns_a_float_32x32_image(name, train):
    out = data.cifar_transform(name, train)(random_image(32))
    assert out.shape == (3, 32, 32)
    assert out.dtype == torch.float32


def test_cifar_eval_transform_only_normalizes():
    """An image filled with the dataset mean colour maps to zero."""
    colour = tuple(round(channel * 255) for channel in CIFAR10_MEAN)
    out = data.cifar_transform("cifar10", train=False)(Image.new("RGB", (32, 32), colour))
    assert out.abs().max() < 0.01


def test_cifar_eval_transform_is_deterministic_and_train_transform_is_not():
    image = random_image(32)
    eval_transform = data.cifar_transform("cifar10", train=False)
    train_transform = data.cifar_transform("cifar10", train=True)
    assert torch.equal(eval_transform(image), eval_transform(image))
    torch.manual_seed(0)
    outputs = [train_transform(image) for _ in range(20)]
    assert any(not torch.equal(outputs[0], other) for other in outputs[1:])


def test_cifar_train_transform_pads_with_zeros_before_cropping():
    """Random crops of a 4-pixel padded image sometimes include the black border."""
    black = tuple(-mean / std for mean, std in zip(CIFAR10_MEAN, data.CIFAR10_STD))
    transform = data.cifar_transform("cifar10", train=True)
    white = Image.new("RGB", (32, 32), (255, 255, 255))
    torch.manual_seed(0)
    has_border = [
        torch.isclose(transform(white)[0], torch.tensor(black[0]), atol=1e-4).any()
        for _ in range(20)
    ]
    assert any(has_border)


def test_train_val_split_is_a_seeded_partition():
    train, val = data.train_val_split(50_000, 5_000, seed=0)
    assert (len(train), len(val)) == (45_000, 5_000)
    assert sorted(train + val) == list(range(50_000))
    assert data.train_val_split(50_000, 5_000, seed=0) == (train, val)
    assert data.train_val_split(50_000, 5_000, seed=1)[1] != val


def test_train_val_split_can_keep_everything_for_training():
    train, val = data.train_val_split(100, 0, seed=0)
    assert (len(train), len(val)) == (100, 0)


def test_cifar_loaders_reject_unknown_names():
    with pytest.raises(ValueError):
        data.cifar_loaders("cifar50", download=False)


@pytest.mark.parametrize(
    "name, folder, num_classes",
    [("cifar10", "cifar-10-batches-py", 10), ("cifar100", "cifar-100-python", 100)],
)
def test_cifar_loaders_on_the_real_dataset(name, folder, num_classes):
    if not (DATA_DIR / folder).is_dir():
        pytest.skip(f"{name} has not been downloaded to {DATA_DIR}")
    loaders = data.cifar_loaders(name, root=str(DATA_DIR), num_workers=0, download=False)
    sizes = {split: len(loader.dataset) for split, loader in loaders.items()}
    assert sizes == {"train": 45_000, "val": 5_000, "test": 10_000, "probe": 128}
    assert len(loaders["test"].dataset.classes) == num_classes

    images, labels = next(iter(loaders["train"]))
    assert images.shape == (128, 3, 32, 32)
    assert labels.shape == (128,)
    assert labels.dtype == torch.long

    # Validation images come from the training set but are not augmented.
    val = loaders["val"].dataset
    assert torch.equal(val[0][0], val[0][0])
    train_indices = set(loaders["train"].dataset.indices)
    assert train_indices.isdisjoint(val.indices)

    # The probe batch: 128 fixed training images, not augmented.
    probe = loaders["probe"]
    assert set(probe.dataset.indices) <= train_indices
    assert len(probe) == 1
    first, second = next(iter(probe)), next(iter(probe))
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    again = data.cifar_loaders(name, root=str(DATA_DIR), num_workers=0, download=False)
    assert again["probe"].dataset.indices == probe.dataset.indices

    everything = data.cifar_loaders(
        name, root=str(DATA_DIR), val_size=0, num_workers=0, download=False
    )
    assert set(everything) == {"train", "test", "probe"}
    assert len(everything["train"].dataset) == 50_000


# ---------------------------------------------------------------------------
# ImageNet
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_imagenet(tmp_path):
    """Two classes with three images each, in the ImageFolder layout."""
    for split in ("train", "val"):
        for class_name in ("n01440764", "n01443537"):
            folder = tmp_path / split / class_name
            folder.mkdir(parents=True)
            for index in range(3):
                Image.new("RGB", (500, 375), (index * 40, 100, 200)).save(folder / f"{index}.jpg")
    return tmp_path


@pytest.mark.parametrize("train", [True, False])
def test_imagenet_transform_returns_a_float_224x224_image(train):
    out = data.imagenet_transform(train)(random_image(300))
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_imagenet_eval_transform_is_resize_256_then_center_crop_224():
    resize, crop = data.imagenet_transform(train=False).transforms[:2]
    assert resize.size == [256]
    assert tuple(crop.size) == (224, 224)


def test_imagenet_loaders_read_an_image_folder(fake_imagenet):
    loaders = data.imagenet_loaders(
        str(fake_imagenet), source="folder", batch_size=4, num_workers=0
    )
    assert set(loaders) == {"train", "val", "probe"}
    # The probe batch: fixed training images with the evaluation transform.
    (probe_images, _), = list(loaders["probe"])
    assert probe_images.shape == (4, 3, 224, 224)
    assert torch.equal(next(iter(loaders["probe"]))[0], probe_images)
    assert len(loaders["train"].dataset) == len(loaders["val"].dataset) == 6
    images, labels = next(iter(loaders["train"]))
    assert images.shape == (4, 3, 224, 224)
    assert set(labels.tolist()) <= {0, 1}


def test_imagenet_loaders_explain_a_missing_folder_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        data.imagenet_loaders(str(tmp_path), source="folder")


def test_imagenet_loaders_reject_unknown_sources(tmp_path):
    with pytest.raises(ValueError):
        data.imagenet_loaders(str(tmp_path), source="kaggle")


def fake_hugging_face_imagenet() -> hf_datasets.DatasetDict:
    """Splits shaped like ILSVRC/imagenet-1k, including non-RGB images."""
    features = hf_datasets.Features(
        {"image": hf_datasets.Image(), "label": hf_datasets.ClassLabel(names=["tench", "goldfish"])}
    )
    images = [
        Image.new("RGB", (500, 375), (10, 100, 200)),
        Image.new("L", (400, 300), 128),  # grayscale
        Image.new("CMYK", (300, 400), (0, 50, 100, 0)),
        Image.new("RGB", (640, 480), (200, 100, 10)),
    ]
    labels = [0, 1, 0, 1]

    def split(split_labels: list[int]) -> hf_datasets.Dataset:
        return hf_datasets.Dataset.from_dict(
            {"image": images, "label": split_labels}, features=features
        )

    return hf_datasets.DatasetDict(
        {"train": split(labels), "validation": split(labels), "test": split([-1] * 4)}
    )


def test_imagenet_loaders_read_the_hugging_face_dataset(monkeypatch, tmp_path):
    calls = []

    def fake_load_dataset(name, cache_dir):
        calls.append((name, cache_dir))
        return fake_hugging_face_imagenet()

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)
    loaders = data.imagenet_loaders(str(tmp_path), batch_size=4, num_workers=0)

    assert calls == [("ILSVRC/imagenet-1k", str(tmp_path))]
    assert set(loaders) == {"train", "val", "probe"}
    probe_images, _ = next(iter(loaders["probe"]))
    assert probe_images.shape == (4, 3, 224, 224)
    assert torch.equal(next(iter(loaders["probe"]))[0], probe_images)
    for split in ("train", "val"):
        images, labels = next(iter(loaders[split]))
        # Grayscale and CMYK images come out with three channels too.
        assert images.shape == (4, 3, 224, 224)
        assert images.dtype == torch.float32
        assert labels.dtype == torch.long
        assert sorted(labels.tolist()) == [0, 0, 1, 1]
    # Validation images are not augmented, so they come back in order and unchanged.
    first, second = next(iter(loaders["val"])), next(iter(loaders["val"]))
    assert torch.equal(first[0], second[0])
    assert first[1].tolist() == [0, 1, 0, 1]


# ---------------------------------------------------------------------------
# Token-based batching
# ---------------------------------------------------------------------------


def random_lengths(count: int, seed: int = 0) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    src = [rng.randint(2, 60) for _ in range(count)]
    tgt = [max(2, length + rng.randint(-5, 5)) for length in src]
    return src, tgt


def test_sampler_uses_every_sentence_exactly_once():
    src, tgt = random_lengths(1000)
    sampler = data.TokenBatchSampler(src, tgt, max_tokens=500)
    assert sorted(index for batch in sampler for index in batch) == list(range(1000))
    assert len(sampler) == len(list(sampler))


def test_sampler_caps_padded_tokens_on_both_sides():
    src, tgt = random_lengths(1000)
    for batch in data.TokenBatchSampler(src, tgt, max_tokens=500):
        assert len(batch) * max(src[i] for i in batch) <= 500
        assert len(batch) * max(tgt[i] for i in batch) <= 500


def test_sampler_groups_sentences_of_similar_length():
    """Sorting by length keeps padding low: most of each batch is real tokens."""
    src, tgt = random_lengths(5000)
    real = padded = 0
    for batch in data.TokenBatchSampler(src, tgt, max_tokens=500):
        real += sum(src[i] for i in batch)
        padded += len(batch) * max(src[i] for i in batch)
    assert real / padded > 0.85


def test_sampler_shuffles_batch_order_each_pass_and_is_seeded():
    src, tgt = random_lengths(1000)
    sampler = data.TokenBatchSampler(src, tgt, max_tokens=500, seed=0)
    first, second = list(sampler), list(sampler)
    assert first != second
    assert sorted(first) == sorted(second)
    assert list(data.TokenBatchSampler(src, tgt, max_tokens=500, seed=0)) == first

    ordered = data.TokenBatchSampler(src, tgt, max_tokens=500, shuffle=False)
    assert list(ordered) == list(ordered)


def test_sampler_can_be_restricted_to_some_sentences():
    src, tgt = random_lengths(100)
    keep = list(range(0, 100, 2))
    sampler = data.TokenBatchSampler(src, tgt, max_tokens=500, indices=keep)
    assert sorted(index for batch in sampler for index in batch) == keep


def test_sampler_rejects_a_sentence_longer_than_max_tokens():
    with pytest.raises(ValueError):
        data.TokenBatchSampler([5, 600], [5, 5], max_tokens=500)


# ---------------------------------------------------------------------------
# Translation pipeline, on a small made-up corpus
# ---------------------------------------------------------------------------

ENGLISH = "the a house garden small green is was in on my your dog cat runs sleeps".split()
GERMAN = "der die das haus garten klein grün ist war in auf mein dein hund katze läuft".split()
VOCAB_SIZE = 300
MAX_LENGTH = 40


def make_pairs(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pairs = []
    for _ in range(count):
        length = rng.randint(1, 10)
        pairs.append(
            {
                "en": " ".join(rng.choices(ENGLISH, k=length)),
                "de": " ".join(rng.choices(GERMAN, k=length)),
            }
        )
    return pairs


@pytest.fixture(scope="module")
def corpus():
    train = make_pairs(300, seed=0)
    train.append({"en": "", "de": "der hund"})  # empty source
    train.append({"en": " ".join(ENGLISH * 20), "de": "der hund"})  # too long
    splits = {"train": train, "validation": make_pairs(20, seed=1), "test": make_pairs(25, seed=2)}
    return hf_datasets.DatasetDict(
        {name: hf_datasets.Dataset.from_dict({"translation": pairs}) for name, pairs in splits.items()}
    )


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("translation")


def build_loaders(corpus, cache_dir, **overrides):
    kwargs = dict(
        src_lang="en",
        tgt_lang="de",
        vocab_size=VOCAB_SIZE,
        max_tokens=200,
        max_length=MAX_LENGTH,
        eval_batch_size=8,
        num_workers=0,
        seed=0,
    )
    kwargs.update(overrides)
    return data.translation_loaders(corpus, cache_dir, **kwargs)


@pytest.fixture(scope="module")
def loaders(corpus, cache_dir):
    return build_loaders(corpus, cache_dir)


@pytest.fixture(scope="module")
def tokenizer(loaders, cache_dir):
    return Tokenizer.from_file(str(data.tokenizer_path(cache_dir, VOCAB_SIZE)))


def test_tokenizer_reserves_the_first_ids_for_special_tokens(tokenizer):
    assert [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] == [PAD_ID, BOS_ID, EOS_ID]
    assert tokenizer.get_vocab_size() <= VOCAB_SIZE


def test_one_tokenizer_round_trips_both_languages(tokenizer):
    for text in ("the small dog sleeps", "der hund läuft auf das grün haus", "Unseen wörds, 123!"):
        ids = tokenizer.encode(text).ids
        assert tokenizer.decode(ids) == text
        assert not set(ids) & {PAD_ID, BOS_ID, EOS_ID}


def test_tokenizer_merges_bytes_into_subwords(tokenizer):
    text = "the small dog sleeps in my garden"
    assert len(tokenizer.encode(text).ids) < len(text.encode())


def test_loaders_cover_train_val_and_test(loaders):
    assert set(loaders) == {"train", "val", "test", "probe"}
    assert len(loaders["val"].dataset) == 20
    assert len(loaders["test"].dataset) == 25


def test_batches_hold_padded_source_and_shifted_target(loaders):
    for split in ("train", "val"):
        for batch in loaders[split]:
            src, tgt_in, tgt_out = batch["src"], batch["tgt_in"], batch["tgt_out"]
            assert src.dtype == tgt_in.dtype == tgt_out.dtype == torch.long
            assert tgt_in.shape == tgt_out.shape
            assert src.shape[0] == tgt_in.shape[0]
            for row in range(src.shape[0]):
                src_len = int((src[row] != PAD_ID).sum())
                tgt_len = int((tgt_out[row] != PAD_ID).sum())
                # Real tokens come first, end with EOS, and padding fills the rest.
                assert src[row, src_len - 1] == EOS_ID
                assert (src[row, src_len:] == PAD_ID).all()
                assert tgt_out[row, tgt_len - 1] == EOS_ID
                assert (tgt_out[row, tgt_len:] == PAD_ID).all()
                # The decoder input is the target shifted right behind BOS.
                assert tgt_in[row, 0] == BOS_ID
                assert torch.equal(tgt_in[row, 1:tgt_len], tgt_out[row, : tgt_len - 1])
                assert (tgt_in[row, tgt_len:] == PAD_ID).all()


def test_batch_decodes_back_to_the_original_sentences(corpus, loaders, tokenizer):
    originals = {(pair["en"], pair["de"]) for pair in corpus["validation"]["translation"]}
    decoded = set()
    for batch in loaders["val"]:
        for src, tgt in zip(batch["src"].tolist(), batch["tgt_out"].tolist()):
            decoded.add((tokenizer.decode(src), tokenizer.decode(tgt)))
    assert decoded == originals


def test_probe_batch_is_fixed_training_pairs_of_mixed_lengths(corpus, loaders, tokenizer):
    probe = loaders["probe"]
    assert len(probe) == 1
    batch = next(iter(probe))
    assert batch["src"].numel() <= 200 and batch["tgt_out"].numel() <= 200
    for key in ("src", "tgt_in", "tgt_out"):
        assert torch.equal(next(iter(probe))[key], batch[key])
    # Random training pairs, not one length bucket like a training batch.
    src_lengths = (batch["src"] != PAD_ID).sum(dim=1)
    assert len(set(src_lengths.tolist())) > 1
    training = {(pair["en"], pair["de"]) for pair in corpus["train"]["translation"]}
    for src, tgt in zip(batch["src"].tolist(), batch["tgt_out"].tolist()):
        assert (tokenizer.decode(src), tokenizer.decode(tgt)) in training


def test_training_batches_respect_the_token_budget(loaders):
    for batch in loaders["train"]:
        assert batch["src"].numel() <= 200
        assert batch["tgt_out"].numel() <= 200


def test_training_skips_empty_and_overlong_pairs_only(corpus, loaders, tokenizer):
    def usable(pair: dict) -> bool:
        lengths = [len(tokenizer.encode(pair[lang]).ids) + 1 for lang in ("en", "de")]
        return min(lengths) > 1 and max(lengths) <= MAX_LENGTH

    pairs = corpus["train"]["translation"]
    expected = sum(usable(pair) for pair in pairs)
    assert expected == len(pairs) - 2
    assert sum(batch["src"].shape[0] for batch in loaders["train"]) == expected


def test_tokenizer_and_tokenized_corpus_are_cached(corpus, cache_dir, loaders):
    path = data.tokenizer_path(cache_dir, VOCAB_SIZE)
    modified = path.stat().st_mtime_ns
    again = build_loaders(corpus, cache_dir)
    assert path.stat().st_mtime_ns == modified
    assert len(again["train"]) == len(loaders["train"])
    assert (cache_dir / f"tokenized-en-de-{VOCAB_SIZE}").is_dir()


def test_batches_feed_the_transformer(loaders, tokenizer):
    model = Transformer(tokenizer.get_vocab_size(), d_model=32, num_heads=2, num_layers=1, d_ff=64)
    batch = next(iter(loaders["train"]))
    logits = model(batch["src"], batch["tgt_in"])
    assert logits.shape == (*batch["tgt_out"].shape, tokenizer.get_vocab_size())
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), batch["tgt_out"].flatten(), ignore_index=PAD_ID
    )
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Hydra configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, function",
    [
        ("cifar10", data.cifar_loaders),
        ("cifar100", data.cifar_loaders),
        ("imagenet", data.imagenet_loaders),
        ("wmt14_en_de", data.wmt14_en_de_loaders),
    ],
)
def test_hydra_data_config_matches_the_loader_function(name, function):
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=[f"data={name}"])
    assert hydra.utils.get_method(cfg.data._target_) is function
    kwargs = {key: value for key, value in cfg.data.items() if key != "_target_"}
    inspect.signature(function).bind(**kwargs)
    if function is data.cifar_loaders:
        assert kwargs["name"] == name
