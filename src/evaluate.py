"""
BLEU(bilingual evaluation understudy) evaluation and comparison with pre-trained model Helsinki-NLP/opus-mt-en-jap
"""

import torch
import sacrebleu
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import random


def compute_bleu(references, hypotheses):
    """
    Calculate BLEU score with sacrebleu
    references: list of strings (target sentences)
    hypotheses: list of strings (predicted sentences)
    """
    # sacrebleu expects list of references per hypothesis
    refs = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(hypotheses, list(zip(*refs)))
    return bleu.score


def translate_with_my_model(model, sp_src, sp_tgt, sentences, device='cuda', max_len=50, bos_id=2, eos_id=3):
    """
    translate using my model
    sentences: list of strings in source language
    """
    model.eval()
    translations = []
    
    with torch.no_grad():
        for sent in tqdm(sentences, desc="My Model"):
            # Tokenize
            src_ids = sp_src.encode(sent, out_type=int, add_bos=True, add_eos=True)
            src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
            
            # Generate
            output = model.translate(src_tensor, max_len=max_len, bos_id=bos_id, eos_id=eos_id)
            
            # Decode (remove BOS/EOS)
            out_ids = output[0].cpu().tolist()
            # Remove special tokens
            out_ids = [id for id in out_ids if id not in [bos_id, eos_id, 0]]
            translated = sp_tgt.decode(out_ids)
            translations.append(translated)
    
    return translations


def translate_with_pretrained(sentences, model_name="Helsinki-NLP/opus-mt-en-jap", direction="en-jp", batch_size=8):
    """
    Translation with pre-trained model MarianMT
    direction: 'en-jp' o 'jp-en'
    """
    if direction == "jp-en":
        model_name = "Helsinki-NLP/opus-mt-ja-en"
    
    print(f"Loading pretrained model: {model_name}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()
    translations = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Pretrained Model"):
            batch = sentences[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            translated = model.generate(**inputs)
            decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
            translations.extend(decoded)
    
    return translations


def evaluate_models(
    my_model,
    sp_en,
    sp_jp,
    test_en_sentences,
    test_jp_sentences,
    direction="en-jp",
    device='cuda'
):
    """
    Complete comparison of 1000 sentences
    direction: 'en-jp' (inglese -> giapponese) o 'jp-en'
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION: {direction.upper()}")
    print(f"{'='*60}")
    
    if direction == "en-jp":
        src_sents = test_en_sentences
        ref_sents = test_jp_sentences
        my_translations = translate_with_my_model(
            my_model, sp_en, sp_jp, src_sents, device=device
        )
        pretrained_translations = translate_with_pretrained(
            src_sents, direction="en-jp"
        )
    else:
        src_sents = test_jp_sentences
        ref_sents = test_en_sentences
        my_translations = translate_with_my_model(
            my_model, sp_jp, sp_en, src_sents, device=device
        )
        pretrained_translations = translate_with_pretrained(
            src_sents, direction="jp-en"
        )
    
    # Calcola BLEU
    my_bleu = compute_bleu(ref_sents, my_translations)
    pretrained_bleu = compute_bleu(ref_sents, pretrained_translations)
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  My Model BLEU:      {my_bleu:.2f}")
    print(f"  Pretrained BLEU:    {pretrained_bleu:.2f}")
    print(f"{'='*60}")
    
    # Salva esempi per ispezione manuale
    results = {
        'source': src_sents,
        'reference': ref_sents,
        'my_translation': my_translations,
        'pretrained_translation': pretrained_translations,
        'my_bleu': my_bleu,
        'pretrained_bleu': pretrained_bleu
    }
    
    return results


def extract_test_set(en_file, jp_file, n=1000, seed=42):
    """
    Estrae n frasi random dal dataset per il test set
    """
    with open(en_file, 'r', encoding='utf-8') as f:
        en_lines = f.readlines()
    with open(jp_file, 'r', encoding='utf-8') as f:
        jp_lines = f.readlines()
    
    # Assicurati siano allineati
    assert len(en_lines) == len(jp_lines)
    
    random.seed(seed)
    indices = random.sample(range(len(en_lines)), n)
    
    test_en = [en_lines[i].strip() for i in indices]
    test_jp = [jp_lines[i].strip() for i in indices]
    
    return test_en, test_jp, indices