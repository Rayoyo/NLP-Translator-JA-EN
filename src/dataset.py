"""
Dataset memory-efficient for 8GB txt dataset
It does NOT load everything in RAM, it reads on-the-fly using offset index
- LazyTranslationDataset: Custom dataset class for fast file position access
- collate_fn: Custom collation function for dynamic padding
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from functools import partial


class LazyTranslationDataset(Dataset):
    """
    Custom dataset class for fast file position access
    """
    def __init__(self, path_en, path_jp, sp_en, sp_jp, max_samples=None):
        self.sp_en = sp_en     # sentencepiece tokenizer for English
        self.sp_jp = sp_jp     # sentencepiece tokenizer for Japanese
        self.path_en = path_en 
        self.path_jp = path_jp

        # Create parallel offset lists for both English and Japanese files
        self.en_offsets = []
        self.jp_offsets = []

        with open(path_en, 'rb') as f_en, open(path_jp, 'rb') as f_jp:
            while True:
                pos_en = f_en.tell()
                pos_jp = f_jp.tell()

                line_en = f_en.readline()
                line_jp = f_jp.readline()

                # if either file has reached EOF, we stop indexing
                if not line_en or not line_jp:
                    break

                self.en_offsets.append(pos_en)
                self.jp_offsets.append(pos_jp)

                if max_samples and len(self.en_offsets) >= max_samples:
                    break

        self.len = len(self.en_offsets)
        print(f"LazyDataset: indexed {self.len:,} lines in parallel.")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Access directly the line in both files using the offset
        with open(self.path_en, 'rb') as f_en, \
             open(self.path_jp, 'rb') as f_jp:
            
            f_en.seek(self.en_offsets[idx])
            f_jp.seek(self.jp_offsets[idx])
            
            en_text = f_en.readline().decode('utf-8', errors='replace').strip()
            jp_text = f_jp.readline().decode('utf-8', errors='replace').strip()

        # Tokenizzazione corretta
        en_ids = self.sp_en.encode(en_text, out_type=int, add_bos=True, add_eos=True)
        jp_ids = self.sp_jp.encode(jp_text, out_type=int, add_bos=True, add_eos=True)

        return torch.tensor(en_ids, dtype=torch.long), torch.tensor(jp_ids, dtype=torch.long)

def collate_fn(batch, pad_idx=0):
    src_batch, tgt_batch = zip(*batch)

    src_max = max(s.size(0) for s in src_batch)  
    tgt_max = max(t.size(0) for t in tgt_batch)  

    src_padded = torch.full((len(src_batch), src_max), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((len(tgt_batch), tgt_max), pad_idx, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, :src.size(0)] = src
        tgt_padded[i, :tgt.size(0)] = tgt

    return src_padded, tgt_padded
    

def create_train_val_dataloaders(path_en, path_jp, sp_en, sp_jp, batch_size=32,
                                 num_workers=2, max_samples=None, train_ratio=0.90):
    """
    Create and return two separate DataLoaders (Train and Validation) using random_split.
    """
    # 1. Complete dataset initialization (with double offset indexing)
    full_dataset = LazyTranslationDataset(path_en, path_jp, sp_en, sp_jp, max_samples)
    
    # 2. Calculate sizes for split (90% e 10%)
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # 3. Execute random_split with a fixed seed for reproducibility
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    collate = partial(collate_fn, pad_idx=0)
    pin_mem = True if torch.cuda.is_available() else False

    # 4. Training loader (with shuffle enabled)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_mem
    )

    # 5. Validation loader (shuffle disabled)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,     # number of samples per batch (32)
        shuffle=False,             # shuffle the data at every epoch to improve training
        num_workers=num_workers,   # number of subprocesses to use for data loading (0 = main process will do the loading)
        collate_fn=collate,        # use our custom collate function for dynamic padding
        pin_memory=pin_mem
    )

    print(f"-> Train Loader ready: {len(train_dataset):,} samples ({len(train_loader)} batches)")
    print(f"-> Val Loader ready: {len(val_dataset):,} samples ({len(val_loader)} batches)")

    return train_loader, val_loader