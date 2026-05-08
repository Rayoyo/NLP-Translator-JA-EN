"""
Training and tokenizer menaging SentencePiece
Training directly on files (not on RAM)
"""

import sentencepiece as spm
import os


def train_tokenizer(file_path, prefix, vocab_size=32000, model_type='bpe', character_coverage=1.0, split_by_whitespace=True):
    
    print(f"Training {prefix} tokenizer on {file_path}...")
    
    spm.SentencePieceTrainer.train(
        input=file_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        split_by_whitespace=split_by_whitespace,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        normalization_rule_name='nmt_nfkc_cf',  # Aggressive for japanese
        remove_extra_whitespaces=True,
        max_sentencepiece_length=16,
        max_sentence_length=2048,
        shuffle_input_sentence=True,
        seed_sentencepiece_size=1000000,
        num_threads=os.cpu_count() or 4
    )
    
    print(f"Saved: {prefix}.model, {prefix}.vocab")
    return f"{prefix}.model"


def load_tokenizer(model_path):
    sp = spm.SentencePieceProcessor(model_file=model_path)
    print(f"Loaded tokenizer: vocab_size={sp.get_piece_size()}")
    return sp


def setup_tokenizers(en_file, jp_file, vocab_size=32000, model_dir="models"):
    """
    Setup completed: train or load toketokenizers
    """
    os.makedirs(model_dir, exist_ok=True)
    
    en_model_path = f"{model_dir}/en_tokenizer.model"
    jp_model_path = f"{model_dir}/jp_tokenizer.model"
    
    # Train English tokenizer (BPE - works well on english)
    if not os.path.exists(en_model_path):
        train_tokenizer(
            en_file, 
            f"{model_dir}/en_tokenizer",
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=1.0,
            split_by_whitespace=True
        )
    
    # Train Japanese tokenizer (Unigram - best for CJK)
    if not os.path.exists(jp_model_path):
        train_tokenizer(
            jp_file,
            f"{model_dir}/jp_tokenizer",
            vocab_size=vocab_size,
            model_type='unigram',
            character_coverage=0.9995,
            split_by_whitespace=False
        )
    
    sp_en = load_tokenizer(en_model_path)
    sp_jp = load_tokenizer(jp_model_path)
    
    return sp_en, sp_jp