"""
GUI semplice con Gradio per tradurre EN <-> JP
"""

import gradio as gr
import torch
import sentencepiece as spm


class TranslatorApp:
    def __init__(self, model, sp_en, sp_jp, device='cuda'):
        self.model = model
        self.sp_en = sp_en
        self.sp_jp = sp_jp
        self.device = device
        self.model.eval()
        
    def translate(self, text, direction):
        if not text or not text.strip():
            return "Enter your text here"
        
        with torch.no_grad():
            if direction == "EN -> JP":
                sp_src = self.sp_en
                sp_tgt = self.sp_jp
            else:
                sp_src = self.sp_jp
                sp_tgt = self.sp_en
            
            # Tokenize
            src_ids = sp_src.encode(text, out_type=int, add_bos=True, add_eos=True)
            src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.device)
            
            # Generate
            output = self.model.translate(
                src_tensor, 
                max_len=100, 
                bos_id=2,  # BOS
                eos_id=3   # EOS
            )
            
            # Decode
            out_ids = output[0].cpu().tolist()
            out_ids = [id for id in out_ids if id not in [0, 2, 3]]  # rimuovi PAD, BOS, EOS
            result = sp_tgt.decode(out_ids)
            
            return result
    
    def launch(self, share=False):
        interface = gr.Interface(
            fn=self.translate,
            inputs=[
                gr.Textbox(
                    lines=3, 
                    placeholder="Enter text to translate...",
                    label="Input Text"
                ),
                gr.Radio(
                    choices=["EN -> JP", "JP -> EN"],
                    value="EN -> JP",
                    label="Direction"
                )
            ],
            outputs=gr.Textbox(
                lines=3,
                label="Translation"
            ),
            title="EN-JP Transformer Translator",
            description="Translate between English and Japanese using a custom Transformer model.",
            examples=[
                ["Hello, how are you?", "EN -> JP"],
                ["今日は良い天気ですね。", "JP -> EN"],
            ]
        )
        
        interface.launch(share=share)


def load_model_for_gui(checkpoint_path, src_vocab, tgt_vocab, device='cuda'):
    """
    Carica modello da checkpoint per la GUI.
    """
    from src.transformer import Transformer
    
    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_ff=2048,
        dropout=0.1
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model