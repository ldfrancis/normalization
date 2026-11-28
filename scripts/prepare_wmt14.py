"""Prepare WMT 2014 English-German before training, for clusters whose compute nodes are offline.

Two stages, which write exactly what ``nstudy.core.data.wmt14_en_de_loaders``
reads, so training then starts without downloading or tokenizing:

1. ``download`` (needs internet: run it on a login node; light on CPU):
   the ``wmt/wmt14`` ``de-en`` corpus from the Hugging Face Hub into ``<root>/raw``.

       python scripts/prepare_wmt14.py download

2. ``tokenize`` (no internet needed; heavy: run it in a CPU job, see
   prepare_wmt14.sh): trains the BPE tokenizer (``<root>/bpe-<vocab>.json``)
   and tokenizes the corpus (``<root>/tokenized-en-de-<vocab>``).

       HF_DATASETS_OFFLINE=1 python scripts/prepare_wmt14.py tokenize

``--root`` and ``--vocab-size`` must match the training run's ``data.root`` and
``data.vocab_size`` (defaults: data/wmt14_en_de, 37000, as in config/data/wmt14_en_de.yaml).
Both stages skip what is already there, so they can be rerun safely.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import datasets as hf_datasets

from nstudy.core import data


def download(root: Path) -> None:
    raw = hf_datasets.load_dataset("wmt/wmt14", "de-en", cache_dir=str(root / "raw"))
    for split, rows in raw.items():
        print(f"{split}: {len(rows):,} sentence pairs")


def tokenize(root: Path, vocab_size: int) -> None:
    # Builds the loaders as training does; on the first call that trains the
    # tokenizer and tokenizes the corpus into root.
    loaders = data.wmt14_en_de_loaders(root=str(root), vocab_size=vocab_size, num_workers=0)
    tokenizer = data.tokenizer_path(root, vocab_size)
    tokenized = root / f"tokenized-en-de-{vocab_size}"
    if not (tokenizer.exists() and tokenized.exists()):
        raise RuntimeError(f"expected {tokenizer} and {tokenized} after tokenizing")
    print(f"tokenizer: {tokenizer}\ntokenized corpus: {tokenized}")
    print(f"training batches per pass (1 GPU): {len(loaders['train']):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=["download", "tokenize"])
    parser.add_argument("--root", type=Path, default=Path("data/wmt14_en_de"))
    parser.add_argument("--vocab-size", type=int, default=37000)
    args = parser.parse_args()
    start = time.perf_counter()
    if args.stage == "download":
        download(args.root)
    else:
        tokenize(args.root, args.vocab_size)
    print(f"{args.stage} done in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
