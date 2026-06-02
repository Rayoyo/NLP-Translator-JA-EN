"""
Dataset memory-efficient for 8GB txt dataset
It does NOT load everything in RAM, it reads on-the-fly using offset index
"""

import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial


class LazyTranslationDataset(Dataset):
    """
    Dataset that does not load file on RAM
    Use 'rb' for index (offset byte) and decode in __getitem__
    """
    def __init__(self, path_en, path_jp, sp_en, sp_jp, max_samples=None):
        self.sp_en = sp_en     # sentencepiece tokenizer for English
        self.sp_jp = sp_jp     # sentencepiece tokenizer for Japanese
        self.path_en = path_en 
        self.path_jp = path_jp

        # Build offset index in BINARY mode for accurate byte offsets
        self.offsets = []
        with open(path_en, 'rb') as f:
            '''
            This loop reads the file line by line in binary mode, storing the byte offset of each line's start.
            It allows the dataset to later seek directly to any line without loading the entire file into memory
            '''

            while True:
                # Store the current byte offset before reading the line
                pos = f.tell()

                # Read the line (in binary mode)
                line = f.readline()


                if not line:
                    break
                # Store the offset of the line start

                self.offsets.append(pos)
                '''
                save only the number, not the content
                used RAM: len(self.offsets) * 8 bytes (size of int) ~ 8 bytes per line
                
                '''

                if max_samples and len(self.offsets) >= max_samples:
                    break
                    # if max_samples is set, stop after indexing that many lines to limit RAM usage during indexing

        self.len = len(self.offsets)
        print(f"LazyDataset: indexed {self.len:,} lines (RAM usage: ~{self.len * 8 / 1024 / 1024:.1f} MB)")
        # RAM usage is minimal since we only store byte offsets (integers) and not the actual text data

    def __len__(self):
        return self.len
    # return the number of lines indexed, which is the length of the dataset - the number of samples available for iteration

    def __getitem__(self, idx):
        # Open in BINARY, seek to exact byte, then decode with error handling
        with open(self.path_en, 'rb') as f_en, \
             open(self.path_jp, 'rb') as f_jp:
            '''
            opens the FRESH files every time an item is requested, 
                seeks to the correct byte offset for that line, and reads it
            This allows the dataset to read only the necessary line into memory, keeping RAM usage low even for large files
            '''

            f_en.seek(self.offsets[idx])
            f_jp.seek(self.offsets[idx])
            # Read raw bytes and decode safely

            # Read raw bytes and decode safely
            en_text = f_en.readline().decode('utf-8', errors='replace').strip()
            jp_text = f_jp.readline().decode('utf-8', errors='replace').strip()
            '''
            readline() reads the line as bytes, decode() converts it to a string, and errors='replace' ensures that any decoding issues don't crash the program but instead replace problematic characters with a placeholder
            strip() removes any leading/trailing whitespace, including newlines
            .decode('utf-8', errors='replace') converts bytes to strings - for decoding errors avoiding crashes 
                    and replacing problematic characters with a placeholder
            '''

        # Tokenize
        en_ids = self.sp_en.encode(en_text, out_type=int, add_bos=True, add_eos=True)
        jp_ids = self.sp_jp.encode(jp_text, out_type=int, add_bos=True, add_eos=True)
        '''
        out_type=int returns token IDs as integers, 
        add_bos=True and add_eos=True add special tokens for beginning and 
            end of sentence, which are important for training the model to understand sentence boundaries    
        '''

        return torch.tensor(en_ids, dtype=torch.long), torch.tensor(jp_ids, dtype=torch.long)
        # Returns tokenized sentences as PyTorch tensors, ready for use in training or evaluation


def collate_fn(batch, pad_idx=0):
    """
    Dynamic padding per batch
    batch: list of (src_tensor, tgt_tensor)
    different legths, so we pad to the max length in the batch
    """
    src_batch, tgt_batch = zip(*batch)
    '''
    zip(*batch) unpacks the list of tuples into two separate lists: one for source tensors and one for target tensors
    batch = [(src1, tgt1), (src2, tgt2), ...]
    zip(*batch) → (src1, src2, ...), (tgt1, tgt2, ...)
    '''

    src_max = max(s.size(0) for s in src_batch)  # find the maximum sequence length in the source batch
    tgt_max = max(t.size(0) for t in tgt_batch)  # find the maximum sequence length in the target batch

    src_padded = torch.full((len(src_batch), src_max), pad_idx, dtype=torch.long)
    #tensor (batch_size, max_seq_len) filled with pad_idx (0) to hold the padded source sequences
    
    tgt_padded = torch.full((len(tgt_batch), tgt_max), pad_idx, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, :src.size(0)] = src
        tgt_padded[i, :tgt.size(0)] = tgt
    # copy the original token IDs into the padded tensors, leaving the rest as pad_idx (0)
    return src_padded, tgt_padded
    # returns the batch of padded source and target tensors, ready for input into the model


def create_dataloaders(path_en, path_jp, sp_en, sp_jp, batch_size=32,
                       num_workers=0, max_samples=None):
    """
    Create DataLoader with lazy loading
    """
    dataset = LazyTranslationDataset(path_en, path_jp, sp_en, sp_jp, max_samples)

    collate = partial(collate_fn, pad_idx=0)
    # create a new function where pad_idx is fixed to 0, so we can pass it to DataLoader without needing to 
    # specify it every time

    loader = DataLoader(
        dataset,
        batch_size=batch_size,    # number of samples per batch (32)
        shuffle=True,             # shuffle the data at every epoch to improve training
        num_workers=num_workers,  # number of subprocesses to use for data loading (0 = main process will do the loading)
        collate_fn=collate,       # use our custom collate function for dynamic padding
        pin_memory=True if torch.cuda.is_available() else False
        # pin_memory=True allows faster data transfer to GPU if available, by keeping the data in page-locked memory
    )

    return loader