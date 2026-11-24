"""Datasets and dataloaders for the normalization study.

Image classification:

* ``cifar_loaders("cifar10" | "cifar100")``: train / val / test loaders with the
  augmentation of He et al. (2016), Section 4.2.
* ``imagenet_loaders(root)``: train / val loaders, from Hugging Face or from image folders.

Translation:

* ``wmt14_en_de_loaders()``: WMT 2014 English-German with a shared BPE vocabulary
  and token-based batching, as in Vaswani et al. (2017), Section 5.1.

Every function returns a dict mapping split names to ``DataLoader`` objects.

Each dict also has ``"probe"``: one fixed batch of training examples without
augmentation, as large as one process's share of a training batch, on which
``nstudy.core.probes`` measures the watched layers. It is not a held-out split.

Training batch sizes (``batch_size``, ``max_tokens``) are global: when training
runs on N processes (see ``nstudy.core.distributed``), each process loads 1/N of
every batch, so the papers' batch sizes hold for any number of GPUs. Evaluation
splits are sharded so that each process scores a different part of the data.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import datasets as hf_datasets
import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler, Subset
from torchvision import datasets as tv_datasets
from torchvision.transforms import v2

from nstudy.core import distributed
from nstudy.core.constants import (
    BOS_ID,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
    EOS_ID,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PAD_ID,
    SPECIAL_TOKENS,
)

Loaders = dict[str, DataLoader]


class ShardSampler(Sampler[int]):
    """Indices ``rank, rank + world_size, ...``: one process's part of an evaluation split.

    Unlike ``DistributedSampler``, no index is repeated to even out the shards,
    so metrics summed over the processes count every example exactly once.
    """

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.indices = range(rank, size, world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def eval_sampler(dataset: Dataset) -> Sampler[int] | None:
    """This process's shard of an evaluation set, or None (everything) in one process."""
    if not distributed.is_distributed():
        return None
    return ShardSampler(len(dataset), distributed.rank(), distributed.world_size())


def make_loader(
    dataset: Dataset, batch_size: int, train: bool, num_workers: int, seed: int = 0
) -> DataLoader:
    """Loader of a training split (shuffled) or an evaluation split (in order).

    For training, ``batch_size`` is the global batch size; each process gets
    ``batch_size / world_size`` examples of it. For evaluation, it is the batch
    size of each process.
    """
    sampler: Sampler[int] | None
    generator = None
    if train:
        batch_size = distributed.per_rank(batch_size)
        if distributed.is_distributed():
            # Same seed on every process, so the shards of a pass do not overlap.
            sampler = DistributedSampler(dataset, shuffle=True, seed=seed, drop_last=True)
        else:
            sampler = None
            generator = torch.Generator().manual_seed(seed)
    else:
        sampler = eval_sampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def probe_loader(
    dataset: Dataset, indices: Sequence[int], batch_size: int, seed: int, collate_fn=None
) -> DataLoader:
    """One fixed batch: ``batch_size`` of ``indices`` chosen at random (seeded) from ``dataset``.

    ``dataset`` must not augment, so the batch is the same every time it is read.
    """
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed + 1))
    chosen = [int(indices[i]) for i in order[: min(batch_size, len(indices))].tolist()]
    return DataLoader(
        Subset(dataset, chosen), batch_size=len(chosen), shuffle=False, collate_fn=collate_fn
    )


def to_normalized_tensor(mean: Sequence[float], std: Sequence[float]) -> list:
    """Transforms that turn a PIL image into a normalized float tensor."""
    return [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]


# ---------------------------------------------------------------------------
# CIFAR-10 and CIFAR-100
# ---------------------------------------------------------------------------

CIFAR_DATASETS = {
    "cifar10": (tv_datasets.CIFAR10, CIFAR10_MEAN, CIFAR10_STD),
    "cifar100": (tv_datasets.CIFAR100, CIFAR100_MEAN, CIFAR100_STD),
}


def cifar_transform(name: str, train: bool) -> v2.Compose:
    """Training: pad 4 pixels per side, random 32x32 crop, random horizontal flip."""
    _, mean, std = CIFAR_DATASETS[name]
    augment = [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()] if train else []
    return v2.Compose([*augment, *to_normalized_tensor(mean, std)])


def train_val_split(size: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Randomly split ``range(size)`` into train and validation indices."""
    if not 0 <= val_size < size:
        raise ValueError(f"val_size must be in [0, {size}), got {val_size}")
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[val_size:], order[:val_size]


def cifar_loaders(
    name: str,
    root: str = "data",
    batch_size: int = 128,
    val_size: int = 5000,
    num_workers: int = 4,
    seed: int = 0,
    download: bool = True,
) -> Loaders:
    """Loaders for CIFAR-10 or CIFAR-100.

    ``batch_size`` is the global training batch size. ``val_size`` training images are held out for validation (the paper uses a
    45k/5k split) and are not augmented. With ``val_size=0`` all 50k images are
    used for training and there is no ``"val"`` loader.
    """
    if name not in CIFAR_DATASETS:
        raise ValueError(f"name must be one of {sorted(CIFAR_DATASETS)}, got {name!r}")
    dataset_cls = CIFAR_DATASETS[name][0]
    train_transform = cifar_transform(name, train=True)
    eval_transform = cifar_transform(name, train=False)

    augmented = dataset_cls(root, train=True, transform=train_transform, download=download)
    plain = dataset_cls(root, train=True, transform=eval_transform)
    test = dataset_cls(root, train=False, transform=eval_transform, download=download)
    train_indices, val_indices = train_val_split(len(augmented), val_size, seed)

    loaders = {
        "train": make_loader(
            Subset(augmented, train_indices), batch_size, True, num_workers, seed
        ),
        "test": make_loader(test, batch_size, False, num_workers),
    }
    if val_size > 0:
        loaders["val"] = make_loader(Subset(plain, val_indices), batch_size, False, num_workers)
    loaders["probe"] = probe_loader(plain, train_indices, distributed.per_rank(batch_size), seed)
    return loaders


# ---------------------------------------------------------------------------
# ImageNet
# ---------------------------------------------------------------------------


def imagenet_transform(train: bool, image_size: int = 224) -> v2.Compose:
    """Training: random resized crop and flip. Evaluation: resize then center crop."""
    if train:
        spatial = [v2.RandomResizedCrop(image_size), v2.RandomHorizontalFlip()]
    else:
        spatial = [v2.Resize(round(image_size / 0.875)), v2.CenterCrop(image_size)]
    return v2.Compose([*spatial, *to_normalized_tensor(IMAGENET_MEAN, IMAGENET_STD)])


class HuggingFaceImages(Dataset):
    """Wrap a Hugging Face image-classification split as ``(image tensor, label)`` pairs."""

    def __init__(self, split: hf_datasets.Dataset, transform: v2.Compose) -> None:
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.split[index]
        # A few ImageNet files are grayscale or CMYK.
        return self.transform(row["image"].convert("RGB")), row["label"]


IMAGENET_HF_NAME = "ILSVRC/imagenet-1k"


def imagenet_loaders(
    root: str = "data/imagenet",
    source: str = "huggingface",
    batch_size: int = 256,
    num_workers: int = 8,
    image_size: int = 224,
    seed: int = 0,
) -> Loaders:
    """Loaders for ImageNet-1k.

    ``source="huggingface"`` loads ``ILSVRC/imagenet-1k`` from the Hugging Face
    Hub, downloading it into ``root`` on first use. The dataset is gated: accept
    its terms on the dataset page and log in (``hf auth login``) first.

    ``source="folder"`` reads image files stored as ``root/train/<class>/*``
    and ``root/val/<class>/*``.

    ImageNet's test labels are not public, so the validation set is the only
    held-out split. ``batch_size`` is the global training batch size.
    """
    train_transform = imagenet_transform(True, image_size)
    eval_transform = imagenet_transform(False, image_size)
    if source == "huggingface":
        splits = hf_datasets.load_dataset(IMAGENET_HF_NAME, cache_dir=root)
        train = HuggingFaceImages(splits["train"], train_transform)
        val = HuggingFaceImages(splits["validation"], eval_transform)
    elif source == "folder":
        root_path = Path(root)
        for split in ("train", "val"):
            if not (root_path / split).is_dir():
                raise FileNotFoundError(
                    f"expected a directory of class folders at {root_path / split}"
                )
        train = tv_datasets.ImageFolder(root_path / "train", train_transform)
        val = tv_datasets.ImageFolder(root_path / "val", eval_transform)
    else:
        raise ValueError(f"source must be 'huggingface' or 'folder', got {source!r}")
    # The training images with the evaluation transform (sharing the file list).
    plain = copy.copy(train)
    plain.transform = eval_transform
    return {
        "train": make_loader(train, batch_size, True, num_workers, seed),
        "val": make_loader(val, batch_size, False, num_workers),
        "probe": probe_loader(plain, range(len(plain)), distributed.per_rank(batch_size), seed),
    }


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def translation_probe_indices(
    src_lengths: np.ndarray,
    tgt_lengths: np.ndarray,
    candidates: np.ndarray,
    max_tokens: int,
    seed: int,
    tries: int = 10_000,
) -> list[int]:
    """Random training pairs (seeded) that fit one batch of ``max_tokens`` padded tokens.

    Unlike a training batch, the pairs are not grouped by length, so the batch
    mixes short and long sentences.
    """
    order = torch.randperm(len(candidates), generator=torch.Generator().manual_seed(seed + 1))
    chosen: list[int] = []
    longest = 0
    for position in order[:tries].tolist():
        index = int(candidates[position])
        length = int(max(src_lengths[index], tgt_lengths[index]))
        if (len(chosen) + 1) * max(longest, length) <= max_tokens:
            chosen.append(index)
            longest = max(longest, length)
    return chosen


def train_bpe_tokenizer(texts: Iterable[str], vocab_size: int) -> Tokenizer:
    """Train a byte-level BPE tokenizer whose first ids are the special tokens.

    Working on bytes means any text can be encoded and decoded back exactly,
    so no unknown token is needed.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def tokenizer_path(cache_dir: str | Path, vocab_size: int) -> Path:
    """Where ``translation_loaders`` stores the tokenizer it trains."""
    return Path(cache_dir) / f"bpe-{vocab_size}.json"


class TokenBatchSampler(Sampler[list[int]]):
    """Batches of sentence pairs of similar length, capped by padded token count.

    Pairs are sorted by length and packed so that a batch holds at most
    ``max_tokens`` source tokens and at most ``max_tokens`` target tokens,
    counting padding. The batches are fixed; ``shuffle`` randomizes their order
    on every pass.

    Args:
        src_lengths: source length of every pair in the dataset.
        tgt_lengths: target length of every pair in the dataset.
        max_tokens: cap on ``batch size x longest sentence`` for each side.
        indices: pairs to use; all of them when None.
        shuffle: visit the batches in a new random order on every pass.
        seed: seed of the shuffling.
        rank, world_size: with several processes, each process takes every
            ``world_size``-th batch of the (identically shuffled) order, and all
            take the same number of batches so they make the same number of
            updates. ``max_tokens`` is then the budget of each process.
    """

    def __init__(
        self,
        src_lengths: Sequence[int],
        tgt_lengths: Sequence[int],
        max_tokens: int,
        indices: Sequence[int] | None = None,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        src = np.asarray(src_lengths)
        tgt = np.asarray(tgt_lengths)
        longest = np.maximum(src, tgt)
        indices = np.arange(len(src)) if indices is None else np.asarray(indices)
        if len(indices) and longest[indices].max() > max_tokens:
            raise ValueError(
                f"a sentence has {longest[indices].max()} tokens, more than max_tokens={max_tokens}"
            )
        order = indices[np.argsort(longest[indices], kind="stable")]

        self.batches: list[list[int]] = []
        batch: list[int] = []
        # Sorted by length, so the pair being added is the longest in the batch.
        for index, length in zip(order.tolist(), longest[order].tolist()):
            if batch and (len(batch) + 1) * length > max_tokens:
                self.batches.append(batch)
                batch = []
            batch.append(index)
        if batch:
            self.batches.append(batch)

        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            order = torch.randperm(len(self.batches), generator=self.generator).tolist()
        else:
            order = list(range(len(self.batches)))
        for index in order[self.rank :: self.world_size][: len(self)]:
            yield self.batches[index]

    def __len__(self) -> int:
        """Batches per pass for this process."""
        return len(self.batches) // self.world_size


def pad(sequences: list[list[int]]) -> Tensor:
    width = max(len(sequence) for sequence in sequences)
    return torch.tensor([s + [PAD_ID] * (width - len(s)) for s in sequences], dtype=torch.long)


def collate_translation(rows: list[dict]) -> dict[str, Tensor]:
    """Pad a list of tokenized pairs into ``src``, ``tgt_in`` and ``tgt_out``.

    Both sides end with the end-of-sentence token. ``tgt_in`` is the target
    shifted right behind a beginning-of-sentence token, so position ``t`` of
    ``tgt_in`` is the decoder input for predicting position ``t`` of ``tgt_out``.
    """
    src = [list(row["src"]) for row in rows]
    tgt = [list(row["tgt"]) for row in rows]
    return {
        "src": pad(src),
        "tgt_in": pad([[BOS_ID] + sentence[:-1] for sentence in tgt]),
        "tgt_out": pad(tgt),
    }


def translation_loaders(
    raw: hf_datasets.DatasetDict,
    cache_dir: str | Path,
    src_lang: str,
    tgt_lang: str,
    vocab_size: int,
    max_tokens: int,
    max_length: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
) -> Loaders:
    """Tokenize a translation corpus and build train / val / test loaders.

    ``raw`` has ``train``, ``validation`` and ``test`` splits whose rows look
    like ``{"translation": {src_lang: str, tgt_lang: str}}``. The tokenizer and
    the tokenized corpus are written to ``cache_dir`` and reused on later calls.

    ``max_tokens`` is the global token budget of a training batch, split evenly
    among the processes. ``eval_batch_size`` is per process.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = tokenizer_path(cache_dir, vocab_size)
    if path.exists():
        tokenizer = Tokenizer.from_file(str(path))
    else:

        def sentences() -> Iterator[str]:
            for batch in raw["train"].iter(batch_size=1000):
                for pair in batch["translation"]:
                    yield pair[src_lang]
                    yield pair[tgt_lang]

        tokenizer = train_bpe_tokenizer(sentences(), vocab_size)
        tokenizer.save(str(path))

    tokenized_dir = cache_dir / f"tokenized-{src_lang}-{tgt_lang}-{vocab_size}"
    if tokenized_dir.exists():
        tokenized = hf_datasets.load_from_disk(str(tokenized_dir))
    else:

        def encode(batch: dict) -> dict:
            pairs = batch["translation"]
            src = tokenizer.encode_batch([pair[src_lang] for pair in pairs])
            tgt = tokenizer.encode_batch([pair[tgt_lang] for pair in pairs])
            src_ids = [encoding.ids + [EOS_ID] for encoding in src]
            tgt_ids = [encoding.ids + [EOS_ID] for encoding in tgt]
            return {
                "src": src_ids,
                "tgt": tgt_ids,
                "src_len": [len(ids) for ids in src_ids],
                "tgt_len": [len(ids) for ids in tgt_ids],
            }

        tokenized = raw.map(encode, batched=True, remove_columns=["translation"], desc="Tokenizing")
        tokenized.save_to_disk(str(tokenized_dir))

    train = tokenized["train"]
    lengths = train.with_format("numpy")
    src_len = np.asarray(lengths["src_len"][:])
    tgt_len = np.asarray(lengths["tgt_len"][:])
    # A length of 1 is an empty sentence: only the end-of-sentence token.
    usable = (np.minimum(src_len, tgt_len) > 1) & (np.maximum(src_len, tgt_len) <= max_length)
    sampler = TokenBatchSampler(
        src_len,
        tgt_len,
        max_tokens // distributed.world_size(),
        indices=np.flatnonzero(usable),
        shuffle=True,
        seed=seed,
        rank=distributed.rank(),
        world_size=distributed.world_size(),
    )

    probe_indices = translation_probe_indices(
        src_len, tgt_len, np.flatnonzero(usable), max_tokens // distributed.world_size(), seed
    )

    def eval_loader(split: str) -> DataLoader:
        return DataLoader(
            tokenized[split],
            batch_size=eval_batch_size,
            sampler=eval_sampler(tokenized[split]),
            collate_fn=collate_translation,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return {
        "train": DataLoader(
            train,
            batch_sampler=sampler,
            collate_fn=collate_translation,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        ),
        "val": eval_loader("validation"),
        "test": eval_loader("test"),
        "probe": DataLoader(
            Subset(train, probe_indices),
            batch_size=len(probe_indices),
            collate_fn=collate_translation,
        ),
    }


def wmt14_en_de_loaders(
    root: str = "data/wmt14_en_de",
    vocab_size: int = 37000,
    max_tokens: int = 25000,
    max_length: int = 256,
    eval_batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 0,
) -> Loaders:
    """Loaders for WMT 2014 English-to-German.

    About 4.5M training pairs, with newstest2013 for validation and
    newstest2014 for testing. Training pairs that are empty or longer than
    ``max_length`` tokens on either side are skipped. The first call downloads
    the corpus, trains the tokenizer and tokenizes everything into ``root``.
    """
    raw = hf_datasets.load_dataset("wmt/wmt14", "de-en", cache_dir=str(Path(root) / "raw"))
    return translation_loaders(
        raw,
        cache_dir=root,
        src_lang="en",
        tgt_lang="de",
        vocab_size=vocab_size,
        max_tokens=max_tokens,
        max_length=max_length,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        seed=seed,
    )
