"""
app.py
------
Gradio web interface for the Japanese–English Transformer translator.

Usage
~~~~~
.. code-block:: bash

    python app.py                  # launches a local Gradio server

The app loads a trained checkpoint from ``checkpoints/best_model.pt`` and the
SentencePiece model from ``data/spm.model``.  Both paths can be overridden via
environment variables ``CHECKPOINT_PATH`` and ``SP_MODEL_PATH``.
"""

from __future__ import annotations

import os
import sys

import gradio as gr
import torch

# Make sure src/ is importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model import Seq2SeqTransformer
from tokenizer import load_tokenizer, encode, decode, BOS_ID, EOS_ID, PAD_ID

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoints/best_model.pt")
SP_MODEL_PATH = os.getenv("SP_MODEL_PATH", "data/spm.model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model: Seq2SeqTransformer | None = None
_sp = None


def _load_resources():
    """Load the tokeniser and model weights (called once at startup)."""
    global _model, _sp

    if not os.path.exists(SP_MODEL_PATH):
        raise FileNotFoundError(
            f"SentencePiece model not found at '{SP_MODEL_PATH}'.\n"
            "Run  python src/tokenizer.py --input data/corpus.txt --model_prefix data/spm  first."
        )
    _sp = load_tokenizer(SP_MODEL_PATH)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at '{CHECKPOINT_PATH}'.\n"
            "Train the model first with  python src/train.py  and make sure a "
            "'best_model.pt' file exists in the checkpoints/ directory."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    config = checkpoint["config"]
    _model = Seq2SeqTransformer(**config).to(DEVICE)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()
    print(f"[app] Model loaded from '{CHECKPOINT_PATH}' (epoch {checkpoint.get('epoch', '?')})")


# ---------------------------------------------------------------------------
# Translation logic
# ---------------------------------------------------------------------------

def translate(japanese_text: str, max_len: int = 100) -> str:
    """Translate a Japanese sentence to English.

    Args:
        japanese_text: Input Japanese sentence.
        max_len: Maximum number of tokens to generate.

    Returns:
        Translated English string, or an error message if resources are missing.
    """
    if _model is None or _sp is None:
        return "⚠️ Model not loaded. Please train the model first (see README)."

    japanese_text = japanese_text.strip()
    if not japanese_text:
        return ""

    src_ids = encode(_sp, japanese_text, add_bos=True, add_eos=True)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)

    tgt_ids = _model.translate(src_tensor, bos_id=BOS_ID, eos_id=EOS_ID, max_len=max_len)
    return decode(_sp, tgt_ids)


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

_EXAMPLES = [
    ["猫が窓の外を見ています。"],
    ["今日は良い天気ですね。"],
    ["私は毎日日本語を勉強しています。"],
    ["新幹線は世界で最も速い列車のひとつです。"],
]

_DESCRIPTION = """
## 🇯🇵 → 🇬🇧  Japanese–English Neural Translator

A sequence-to-sequence Transformer trained from scratch on a parallel JA–EN corpus.

**How to use:** Type (or paste) a Japanese sentence in the box on the left and press
**Translate**.  Click one of the example sentences to try it immediately.
"""

_ARTICLE = """
### About this project
Built with [PyTorch](https://pytorch.org/) and [SentencePiece](https://github.com/google/sentencepiece).  
Source code: `src/model.py`, `src/tokenizer.py`, `src/train.py`.
"""


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="JA→EN Translator") as demo:
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            with gr.Column():
                src_box = gr.Textbox(
                    label="Japanese input (日本語)",
                    placeholder="日本語のテキストをここに入力してください…",
                    lines=4,
                )
                max_len_slider = gr.Slider(
                    minimum=20,
                    maximum=200,
                    value=100,
                    step=10,
                    label="Max output tokens",
                )
                translate_btn = gr.Button("Translate", variant="primary")

            with gr.Column():
                tgt_box = gr.Textbox(label="English output", lines=4, interactive=False)

        translate_btn.click(
            fn=translate,
            inputs=[src_box, max_len_slider],
            outputs=tgt_box,
        )
        src_box.submit(
            fn=translate,
            inputs=[src_box, max_len_slider],
            outputs=tgt_box,
        )

        gr.Examples(examples=_EXAMPLES, inputs=src_box)
        gr.Markdown(_ARTICLE)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        _load_resources()
    except FileNotFoundError as exc:
        print(f"[app] WARNING: {exc}")
        print("[app] The interface will still launch but translations will not work.")

    demo = build_interface()
    demo.launch(share=False)
