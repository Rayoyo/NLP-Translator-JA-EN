"""
Training loop with checkpointing, mixed precision and save on Drive
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import time


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        optimizer=None,
        criterion=None,
        scheduler=None,
        device='cuda',
        save_dir='models',
        log_interval=100
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.log_interval = log_interval
        
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(), 
            lr=1e-4, 
            betas=(0.9, 0.98), 
            eps=1e-9
        )
        
        self.criterion = criterion or nn.CrossEntropyLoss(ignore_index=0)  # pad_idx=0
        
        self.scheduler = scheduler
        
        self.scaler = GradScaler()  # Mixed precision
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        os.makedirs(save_dir, exist_ok=True)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (src, tgt) in enumerate(pbar):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # tgt_input: all tokens except last (teacher forcing)
            # tgt_output: all tokens except first (target to predict)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward
            with autocast():
                output = self.model(src, tgt_input)
                # output: (batch, tgt_len-1, vocab_size)
                # tgt_output: (batch, tgt_len-1)
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_output.reshape(-1)
                )
            
            # Backward with scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Stats (excluding padding)
            non_pad_mask = (tgt_output != 0)
            n_tokens = non_pad_mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            
            self.global_step += 1
            
            # Logging
            if batch_idx % self.log_interval == 0:
                avg_loss = total_loss / max(total_tokens, 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        epoch_loss = total_loss / max(total_tokens, 1)
        return epoch_loss
    
    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return None
            
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        for src, tgt in tqdm(self.val_loader, desc="Validation"):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            with autocast():
                output = self.model(src, tgt_input)
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_output.reshape(-1)
                )
            
            non_pad_mask = (tgt_output != 0)
            n_tokens = non_pad_mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
        
        val_loss = total_loss / max(total_tokens, 1)
        return val_loss
    
    def save_checkpoint(self, filename=None, is_best=False):
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}.pt"
        
        path = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model saved!")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.global_step = checkpoint['global_step']
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def fit(self, n_epochs, resume_from=None):
        if resume_from:
            self.load_checkpoint(resume_from)
        
        for epoch in range(self.epoch, n_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Save
            is_best = val_loss is not None and val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(is_best=is_best)
            
            # Print epoch summary
            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch} | Time: {elapsed:.1f}s | Train Loss: {train_loss:.4f}", end="")
            if val_loss:
                print(f" | Val Loss: {val_loss:.4f}")
            else:
                print()
            
            # Auto-save every 2 epochs (for work on Colab)
            if epoch % 2 == 0:
                self.save_checkpoint(f"backup_epoch_{epoch}.pt")


def get_scheduler(optimizer, d_model, warmup_steps=4000):
    """
    Learning rate scheduler for Transformer 
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
    def lr_lambda(step):
        step = max(1, step)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)