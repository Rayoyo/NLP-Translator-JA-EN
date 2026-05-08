#!/usr/bin/env python3
"""
Entry point to start the translator's GUI
Load the trained model and launch Gradio
"""

import torch
from src.transformer import Transformer
from src.gui import TranslatorApp
import sentencepiece as spm
import argparse


def main():
    parser = argparse.ArgumentParser(description='EN-JP Translator GUI')
    parser.add_argument('--checkpoint', type=str, default='models/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--en-tokenizer', type=str, default='models/en_tokenizer.model')
    parser.add_argument('--jp-tokenizer', type=str, default='models/jp_tokenizer.model')
    parser.add_argument('--vocab-size', type=int, default=32000)
    parser.add_argument('--share', action='store_true', help='Create public Gradio link')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizers
    print("Loading tokenizers...")
    sp_en = spm.SentencePieceProcessor(model_file=args.en_tokenizer)
    sp_jp = spm.SentencePieceProcessor(model_file=args.jp_tokenizer)
    
    # Load model
    print("Loading model...")
    model = Transformer(
        src_vocab_size=args.vocab_size,
        tgt_vocab_size=args.vocab_size,
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_ff=2048,
        dropout=0.1
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("Model loaded successfully!")
    
    # Launch GUI
    app = TranslatorApp(model, sp_en, sp_jp, device=device)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()