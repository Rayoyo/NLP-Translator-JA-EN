"""
Dataset memory-efficient for 8GB txt dataset
It does NOT load everything in RAM, it reads on-the-fly using offset index
"""

import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import os


class LazyTranslationDataset(Dataset):
    """
    It builds and offset index (byte position) for every row, 
    then it reads only the current batch from the disk
    """
    def __init__(self, path_en, path_jp, sp_en, sp_jp, max_samples=None):
        self.sp_en = sp_en
        self.sp_jp = sp_jp
        self.path_en = path_en
        self.path_jp = path_jp
        
        # Build offset index (only numbers)
        self.offsets = []
        with open(path_en, 'r', encoding='utf-8') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                self.offsets.append(pos)
                if max_samples and len(self.offsets) >= max_samples:
                    break
        
        # Keep files open for fast seeking (one per worker issue handled in __getitem__)
        self.len = len(self.offsets)
        print(f"LazyDataset: indexed {self.len:,} lines (RAM usage: ~{self.len * 8 / 1024 / 1024:.1f} MB)")
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # Open files fresh (thread-safe for DataLoader num_workers > 0)
    with open(self.path_en, 'r', encoding='utf-8', errors='replace') as f_en, \
         open(self.path_jp, 'r', encoding='utf-8', errors='replace') as f_jp:
        
        f_en.seek(self.offsets[idx])
        f_jp.seek(self.offsets[idx])
        
        en_text = f_en.readline().strip()
        jp_text = f_jp.readline().strip()
    
    # Tokenize
    en_ids = self.sp_en.encode(en_text, out_type=int, add_bos=True, add_eos=True)
    jp_ids = self.sp_jp.encode(jp_text, out_type=int, add_bos=True, add_eos=True)
    
    return torch.tensor(en_ids, dtype=torch.long), torch.tensor(jp_ids, dtype=torch.long)


def collate_fn(batch, pad_idx=0):
    """
    Dynamic padding for batch
    batch: list of (src, tgt) tuples
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Find max lengths
    src_max = max(s.size(0) for s in src_batch)
    tgt_max = max(t.size(0) for t in tgt_batch)
    
    # Pad
    src_padded = torch.full((len(src_batch), src_max), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((len(tgt_batch), tgt_max), pad_idx, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, :src.size(0)] = src
        tgt_padded[i, :tgt.size(0)] = tgt
    
    return src_padded, tgt_padded


def create_dataloaders(path_en, path_jp, sp_en, sp_jp, batch_size=32, num_workers=2, max_samples=None):
    """
    Create DataLoader with lazy loading
    """
    dataset = LazyTranslationDataset(path_en, path_jp, sp_en, sp_jp, max_samples)
    
    # Use partial to pass pad_idx to collate_fn
    from functools import partial
    collate = partial(collate_fn, pad_idx=0)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return loader


class HuggingFaceLazyDataset(Dataset):
    """
    ALTERNATIV: Uses HuggingFace's dataset for memory-mapping
    More efficient, mostly if it converts the format Arrow/Parquet
    """
    def __init__(self, path_en, path_jp, sp_en, sp_jp):
        from datasets import Dataset as HFDataset
        
        # Create dataset HuggingFace from txt file (memory-mapped)
        with open(path_en, 'r', encoding='utf-8') as f:
            en_lines = [line.strip() for line in f]
        with open(path_jp, 'r', encoding='utf-8') as f:
            jp_lines = [line.strip() for line in f]
            
        self.data = HFDataset.from_dict({
            'en': en_lines,
            'jp': jp_lines
        })
        self.sp_en = sp_en
        self.sp_jp = sp_jp
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        en_ids = self.sp_en.encode(item['en'], out_type=int, add_bos=True, add_eos=True)
        jp_ids = self.sp_jp.encode(item['jp'], out_type=int, add_bos=True, add_eos=True)
        return torch.tensor(en_ids), torch.tensor(jp_ids)