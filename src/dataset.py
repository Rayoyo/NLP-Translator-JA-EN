"""
dataset.py
----------
PyTorch Dataset and DataLoader utilities for the Japanese–English
translation corpus.

Each sample in the dataset is a pair ``(src_ids, tgt_ids)`` where:

* ``src_ids`` – source (Japanese) token IDs with BOS and EOS tokens.
* ``tgt_ids`` – target (English) token IDs with BOS and EOS tokens.

The :func:`collate_fn` pads variable-length sequences within a batch so
they can be stacked into fixed-shape tensors.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from tokenizer import encode, load_tokenizer, PAD_ID


class TranslationDataset(Dataset):
    """Dataset for parallel Japanese–English sentence pairs.

    Args:
        src_file: Path to the source-language (Japanese) text file,
                  one sentence per line.
        tgt_file: Path to the target-language (English) text file,
                  one sentence per line.
        sp_model: Path to the shared SentencePiece ``*.model`` file.
        max_len: Maximum number of tokens per sentence (longer sentences
                 are silently discarded, default 128).
    """

    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        sp_model: str,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.sp = load_tokenizer(sp_model)

        src_sentences = self._read_file(src_file)
        tgt_sentences = self._read_file(tgt_file)

        if len(src_sentences) != len(tgt_sentences):
            raise ValueError(
                f"Source and target files must have the same number of lines "
                f"(got {len(src_sentences)} vs {len(tgt_sentences)})."
            )

        self.pairs: List[Tuple[List[int], List[int]]] = []
        for src, tgt in zip(src_sentences, tgt_sentences):
            src_ids = encode(self.sp, src, add_bos=True, add_eos=True)
            tgt_ids = encode(self.sp, tgt, add_bos=True, add_eos=True)
            # Skip sentences that exceed max_len
            if len(src_ids) <= max_len and len(tgt_ids) <= max_len:
                self.pairs.append((src_ids, tgt_ids))

    @staticmethod
    def _read_file(path: str) -> List[str]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        with open(path, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        src_ids, tgt_ids = self.pairs[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch: List[Tuple[Tensor, Tensor]], pad_id: int = PAD_ID):
    """Pad a list of ``(src, tgt)`` tensor pairs to the same length.

    This function is passed as the ``collate_fn`` argument of
    :class:`torch.utils.data.DataLoader`.

    Args:
        batch: List of ``(src_ids, tgt_ids)`` tensor tuples from the dataset.
        pad_id: Token ID used for padding (default: 0).

    Returns:
        Tuple of ``(src_batch, tgt_batch)`` tensors, both of shape
        ``(batch_size, max_len_in_batch)``.
    """
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)
    return src_padded, tgt_padded


def get_dataloader(
    src_file: str,
    tgt_file: str,
    sp_model: str,
    batch_size: int = 32,
    max_len: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Convenience wrapper that builds a :class:`TranslationDataset` and wraps
    it in a :class:`~torch.utils.data.DataLoader`.

    Args:
        src_file: Source (Japanese) text file.
        tgt_file: Target (English) text file.
        sp_model: SentencePiece model file (``*.model``).
        batch_size: Samples per batch.
        max_len: Maximum token length per sentence.
        shuffle: Whether to shuffle the dataset each epoch.
        num_workers: Number of worker processes for loading.

    Returns:
        A configured :class:`~torch.utils.data.DataLoader`.
    """
    dataset = TranslationDataset(src_file, tgt_file, sp_model, max_len=max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
