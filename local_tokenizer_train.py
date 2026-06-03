#!/usr/bin/env python3
"""
Train SentencePiece tokenizers locally (RAM problems on Colab)
Generates: en_tokenizer.model, en_tokenizer.vocab, jp_tokenizer.model, jp_tokenizer.vocab
NOTE: This script is designed to be run on a local machine with sufficient RAM (16-20GB) due to the large dataset size (8GB)
      If you have limited RAM, it will sample 2 million lines from each file (configurable) to reduce memory usage while maintaining tokenizer quality
"""

import os
import random
import shutil
from pathlib import Path
import sentencepiece as spm


# ==================== CONFIGURATION ====================

# Clear file path
EN_FILE = Path("data/processed/english.txt")
JP_FILE = Path("data/processed/japanese.txt")

# Folder to save tokenizers (will be created if not exist)
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Vocab size for both tokenizers (can be adjusted, but 32k is a common choice for NMT)
VOCAB_SIZE = 32000

# How many lines to sample for training (to reduce RAM usage)
# None = uses all lines, requires ~16-20GB RAM
# 2000000 = 2 million lines (advised 2GB RAM, identical quality)
SAMPLE_LINES = 2000000

# Seed for reproducibility (only relevant if SAMPLE_LINES is set)
SEED = 42

# ======================================================

# Utility functions
def count_lines(file_path):
    """Efficient rows count for large files (RAM-friendly)"""
    print(f"CCounting rows in {file_path.name}...")
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count

# Create a sample file with n random lines (if needed)
def create_sample(input_path, output_path, n_lines, seed=42):
    """
    Create a sample file with n random lines
    If the file has fewer lines than n, copy everything
    """
    total = count_lines(input_path)
    print(f"  Total rows: {total:,}")

    # If the file is small, just copy it
    if total <= n_lines:
        print(f"Small file, I use all the lines (total {total:,} <= {n_lines:,})")
        shutil.copy(input_path, output_path)
        return output_path

    # Sample n random lines
    print(f"  Sampling {n_lines:,} random rows...")
    random.seed(seed)
    selected = set(random.sample(range(total), n_lines))

    written = 0

    # Read the input file and write only the selected lines to the output file
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        # Process the file line by line to avoid memory issues
        for i, line in enumerate(fin):
            if i in selected:
                fout.write(line)
                written += 1
            if i % 500_000 == 0 and i > 0:
                print(f"    Processated {i:,} rows...")

    # Final check
    print(f"  Sample created: {written:,} rows -> {output_path}")
    return output_path

def train_tokenizer(input_file, prefix, vocab_size, model_type, character_coverage, split_by_whitespace):
    """
    Train a SentencePiece tokenizer with given parameters
    """
    print(f"\n{'='*60}")
    print(f"Addestramento: {prefix}")
    print(f"  Input: {input_file}")
    print(f"  Model type: {model_type}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Character coverage: {character_coverage}")
    print(f"{'='*60}")

    # Train the tokenizer
    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,

        # special IDs (must be the same in Colab)
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,

        # Other options
        split_by_whitespace=split_by_whitespace,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=True,
        normalization_rule_name="nmt_nfkc_cf",
        max_sentencepiece_length=16,
        max_sentence_length=2048,
        shuffle_input_sentence=True,
        seed_sentencepiece_size=1000000,
        num_threads=os.cpu_count() or 4,
        # Logging
        train_extremely_large_corpus=True if count_lines(input_file) > 1_000_000 else False
    )

    model_file = Path(f"{prefix}.model")
    vocab_file = Path(f"{prefix}.vocab")
    
    print(f"   Completed!")
    print(f"   Model: {model_file} ({model_file.stat().st_size / 1e6:.1f} MB)")
    print(f"   Vocab: {vocab_file}")
    
    return model_file


def verify_tokenizer(model_path, test_sentences, name):
    """Verify that the tokenizer works correctly."""
    print(f"\n--- Verify {name} ---")
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    print(f"Vocabulary: {sp.get_piece_size():,} tokens")

    # Test encoding and decoding
    for sent in test_sentences:
        ids = sp.encode(sent, out_type=int, add_bos=True, add_eos=True)
        decoded = sp.decode(ids)
        print(f"  IN:  {sent}")
        print(f"  IDS: {ids[:15]}{'...' if len(ids) > 15 else ''}")
        print(f"  OUT: {decoded}")
        print()

# Main function
def main():
    print("="*60)
    print("LOCAL TOKENIZER TRAINING - START")
    print("="*60)
    
    # Verify that input files exist
    if not EN_FILE.exists():
        raise FileNotFoundError(f"File not found: {EN_FILE}")
    if not JP_FILE.exists():
        raise FileNotFoundError(f"File not found: {JP_FILE}")

    # Temp file for sampling (if needed)
    tmp_dir = Path("tmp_tokenizer")
    tmp_dir.mkdir(exist_ok=True)

    try:
        # --- ENGLISH TOKENIZER (BPE) ---
        if SAMPLE_LINES:
            en_sample = tmp_dir / "en_sample.txt"
            create_sample(EN_FILE, en_sample, SAMPLE_LINES, SEED)
            en_input = en_sample
        else:
            en_input = EN_FILE

        en_prefix = OUTPUT_DIR / "en_tokenizer"
        
        '''
        For English, we use BPE with 32k vocab size, 100% character coverage, and split by whitespace
        This is a common configuration for NMT tasks and should work well for English text
        '''
        train_tokenizer(
            input_file=en_input,
            prefix=en_prefix,
            vocab_size=VOCAB_SIZE,
            model_type="bpe",
            character_coverage=1.0,
            split_by_whitespace=True
        )

        # --- JAPANESE TOKENIZER (Unigram) ---
        if SAMPLE_LINES:
            jp_sample = tmp_dir / "jp_sample.txt"
            create_sample(JP_FILE, jp_sample, SAMPLE_LINES, SEED)
            jp_input = jp_sample
        else:
            jp_input = JP_FILE

        # For Japanese, we use Unigram with 32k vocab size, 99.95% character coverage, and no whitespace splitting
        # This configuration is often recommended for languages with large character sets like Japanese to ensure good coverage
        jp_prefix = OUTPUT_DIR / "jp_tokenizer"
        train_tokenizer(
            input_file=jp_input,
            prefix=jp_prefix,
            vocab_size=VOCAB_SIZE,
            model_type="unigram",
            character_coverage=0.9995,
            split_by_whitespace=False
        )

        # --- Verify Tokenizers ---
        print(f"\n{'='*60}")
        print("VERIFY TOKENIZERS")
        print(f"{'='*60}")

        # Test sentences for verification (should be representative of the training data)
        verify_tokenizer(
            OUTPUT_DIR / "en_tokenizer.model",
            ["Hello, how are you?", "I love machine learning.", "Thank you very much!"],
            "English"
        )

        # For Japanese, we use some common sentences that should be in the training data
        verify_tokenizer(
            OUTPUT_DIR / "jp_tokenizer.model",
            ["今日は良い天気ですね。", "ありがとうございます。", "機械学習が大好きです。"],
            "Japanese"
        )

        # Final report
        print(f"\n{'='*60}")
        print("ALL SET!")
        print(f"{'='*60}")
        print(f"Files generated in: {OUTPUT_DIR.absolute()}")
        print(f"  - en_tokenizer.model")
        print(f"  - en_tokenizer.vocab")
        print(f"  - jp_tokenizer.model")
        print(f"  - jp_tokenizer.vocab")
        print(f"\nNow upload these 4 files to Google Drive in:")
        print(f"  MyDrive/University/Project-NLP_Translator/models/")

    finally:
        # Cleanup temp files
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
            print(f"\nCleanup completed (removed {tmp_dir})")

# Entry point
if __name__ == "__main__":
    main()