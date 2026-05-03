"""
tokenizer.py
------------
Train and load a shared SentencePiece BPE tokenizer for the
Japanese–English translation model.

Usage (training):
    python src/tokenizer.py \
        --input  data/corpus.txt \
        --model_prefix data/spm \
        --vocab_size 16000

Usage (quick test):
    python src/tokenizer.py --test --model_prefix data/spm
"""

import argparse
import os
import sentencepiece as spm


# Special-token IDs used throughout the project
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def train_tokenizer(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 16000,
    character_coverage: float = 0.9995,
) -> None:
    """Train a SentencePiece BPE tokenizer.

    Args:
        input_file: Path to a plain-text corpus (one sentence per line).
                    For a shared JA/EN vocabulary, concatenate both sides
                    into a single file before calling this function.
        model_prefix: Prefix for the output ``*.model`` and ``*.vocab`` files.
        vocab_size: Total vocabulary size (default 16 000).
        character_coverage: Coverage for characters; keep at ~1.0 for CJK.
    """
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="bpe",
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
    )
    print(
        f"[tokenizer] Tokenizer trained.\n"
        f"  Model : {model_prefix}.model\n"
        f"  Vocab : {model_prefix}.vocab\n"
        f"  Vocab size : {vocab_size}"
    )


def load_tokenizer(model_path: str) -> spm.SentencePieceProcessor:
    """Load a trained SentencePiece tokenizer from *model_path*.

    Args:
        model_path: Path to the ``*.model`` file produced by :func:`train_tokenizer`.

    Returns:
        A ready-to-use :class:`spm.SentencePieceProcessor` instance.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def encode(sp: spm.SentencePieceProcessor, text: str, add_bos: bool = True, add_eos: bool = True):
    """Encode *text* to a list of integer token IDs.

    Args:
        sp: Loaded :class:`spm.SentencePieceProcessor`.
        text: Input string.
        add_bos: Prepend BOS token.
        add_eos: Append EOS token.

    Returns:
        List of integer token IDs.
    """
    ids = sp.encode(text, out_type=int)
    if add_bos:
        ids = [BOS_ID] + ids
    if add_eos:
        ids = ids + [EOS_ID]
    return ids


def decode(sp: spm.SentencePieceProcessor, ids) -> str:
    """Decode a list of integer token IDs back to a string.

    Args:
        sp: Loaded :class:`spm.SentencePieceProcessor`.
        ids: Iterable of integer token IDs.

    Returns:
        Decoded string.
    """
    # Filter out special tokens before decoding
    ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
    return sp.decode(ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="SentencePiece tokenizer trainer / tester")
    parser.add_argument("--input", type=str, help="Input corpus file for training")
    parser.add_argument(
        "--model_prefix",
        type=str,
        default="data/spm",
        help="Output model prefix (default: data/spm)",
    )
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Load an existing model and run a quick encode/decode test",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.test:
        model_path = args.model_prefix + ".model"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        sp = load_tokenizer(model_path)
        samples = [
            "猫が窓の外を見ています。",
            "The cat is looking out the window.",
        ]
        for text in samples:
            ids = encode(sp, text)
            decoded = decode(sp, ids)
            print(f"Input   : {text}")
            print(f"Encoded : {ids}")
            print(f"Decoded : {decoded}")
            print()
    else:
        if args.input is None:
            raise ValueError("--input is required when not using --test")
        train_tokenizer(
            args.input,
            args.model_prefix,
            args.vocab_size,
            args.character_coverage,
        )
