"""
Training loop with checkpointing, mixed precision and save on Drive
"""

import math

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import time


class Trainer:
    def __init__(
        self,
        model,                # transformer instance
        train_loader,         # DataLoader for training
        val_loader=None,      # DataLoader for validation (optional)
        optimizer=None,       # optimizer (default: Adam with transformer-friendly settings)
        criterion=None,       # loss function (default: CrossEntropyLoss with ignore_index=0 for padding)
        scheduler=None,       # learning rate scheduler (optional)
        device='cuda',
        save_dir='models',    # directory to save checkpoints
        log_interval=100,     # print training stats every N batches
        accumulator_steps=4  # number of steps to accumulate gradients for (default: 1 = no accumulation)
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.accumulator_steps = accumulator_steps
        # default optimizer: Adam with transformer-friendly settings
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(), 
            lr=1e-4,             # initial learning rate (will be overridden by scheduler if provided)
            betas=(0.9, 0.98),   # momentum adam (0.9 for gradient, 0.98 for squared gradient)
            eps=1e-9             # epsilon for numerical stability
        )
        
        # default criterion: CrossEntropyLoss with ignore_index=0 for padding
        self.criterion = criterion or nn.CrossEntropyLoss(ignore_index=0)  # pad_idx=0
        
        self.scheduler = scheduler
        
        self.scaler = GradScaler()         # Mixed precision
        self.global_step = 0               # counts total optimization steps (for scheduler)
        self.epoch = 0                     # current epoch
        self.best_val_loss = float('inf')  # best validation loss for checkpointing
        
        os.makedirs(save_dir, exist_ok=True)
        
    def train_epoch(self):
        self.model.train()  # training mode: enables dropout, batch norm & uses batch stats
        total_loss = 0
        total_tokens = 0
        # counters for avg loss weight by number of tokens (excluding padding)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        # tqdm progress bar for training batches

        self.optimizer.zero_grad()  # reset gradients at the start of the epoch (or after accumulation steps)
        
        for batch_idx, (src, tgt) in enumerate(pbar):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # tgt_input: all tokens except last (teacher forcing)
            # tgt_output: all tokens except first (target to predict)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # self.optimizer.zero_grad()  # reset gradients
            
            # Mixed precision forward
            with autocast():
                output = self.model(src, tgt_input)
                # Forward pass: (batch, tgt_len-1, vocab_size)
                # tgt_output: (batch, tgt_len-1)
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    # .reshape(-1, 32000): flatten (batch*(tgt_len-1), vocab_size)

                    tgt_output.reshape(-1)
                    # flatten (batch*(tgt_len-1))
                )
            
            # Backward with scaling
            self.scaler.scale(loss).backward()
            # .scale(loss): scales the loss for better precision in gradients
            # .backward(): computes gradients (scaled)

            # Stats: use loss.item() original (NOT scaled) for correct log
            non_pad_mask = (tgt_output != 0)             # mask to count only non-padding tokens
            n_tokens = non_pad_mask.sum().item()         # number of non-padding tokens in the batch
            total_loss += loss.item() * n_tokens         # accumulate total loss weighted by number of tokens (excluding padding)
            total_tokens += n_tokens                     # accumulate total tokens (excluding padding) for avg loss calculation
        

            # weight update and gradient clipping every accumulation_steps
            if (batch_idx + 1) % self.accumulator_steps == 0:
                self.scaler.unscale_(self.optimizer)    # remove scaling for clipping
                # clip gradients to prevent explosion (max norm of 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)    # update parameters with scaled gradients
                self.scaler.update()                # update the scale factor for next iteration (batch)
                self.optimizer.zero_grad()          # reset gradients after update
                
                if self.scheduler:
                    # update learning rate according to scheduler (usually per optimization step)
                    self.scheduler.step()
                self.global_step += 1          

            # Logging
            if batch_idx % self.log_interval == 0:
                avg_loss = total_loss / max(total_tokens, 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'step': f'{batch_idx}/{len(self.train_loader)}'
                })

        # Manage remaning batchs
        if (len(self.train_loader) % self.accumulator_steps) != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()
            self.global_step += 1
        
        epoch_loss = total_loss / max(total_tokens, 1)
        return epoch_loss
    # avg loss for the epoch weighted by number of tokens (excluding padding)
    
    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return None
            
        self.model.eval()  # evaluation mode: disables dropout, batch norm uses running stats
        total_loss = 0      
        total_tokens = 0  
        
        # tqdm progress bar for validation batches
        for src, tgt in tqdm(self.val_loader, desc="Validation"):
            src = src.to(self.device)  
            tgt = tgt.to(self.device)
            
            tgt_input = tgt[:, :-1]  # all tokens except last (teacher forcing)
            tgt_output = tgt[:, 1:]  # all tokens except first (target to predict)
            
            # Mixed precision forward
            with autocast():
                output = self.model(src, tgt_input)
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_output.reshape(-1)  
                )
            
            # Stats (excluding padding)
            non_pad_mask = (tgt_output != 0)
            n_tokens = non_pad_mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
        
        # average validation loss weighted by number of tokens (excluding padding)
        val_loss = total_loss / max(total_tokens, 1)
        return val_loss
    
    # Checkpointing
    def save_checkpoint(self, filename=None, is_best=False):
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}.pt"
        
        # Save checkpoint with model state, optimizer state, scaler state, epoch, best val loss and global step
        path = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),          # model parameters
            'optimizer_state_dict': self.optimizer.state_dict(),  # optimizer parameters (including learning rate)
            'scaler_state_dict': self.scaler.state_dict(),        # scaler parameters for mixed precision      
            'best_val_loss': self.best_val_loss,                  # best validation loss for checkpointing
            'global_step': self.global_step,                      # global step count for scheduler and logging
        }
        
        # Save checkpoint to disk
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model saved!")
    
    # Load checkpoint (for resuming training)
    def load_checkpoint(self, path):
        # Load checkpoint from disk and restore model state, optimizer state, scaler state, epoch, 
        # best val loss and global step
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.global_step = checkpoint['global_step']
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    # Main training loop
    def fit(self, n_epochs, resume_from=None):
        '''
        Main training loop with optional resuming from checkpoint.
        For each epoch:
        - Train for one epoch and calculate training loss
        - Validate and calculate validation loss (if val_loader is provided)
        - Save checkpoint (and best model if validation loss improved)
        - Print epoch summary with time, training loss and validation loss
        NOTE: checkpoint epoch 0 saves as self.epoch = 0, so range (0, 10) will restart from epoch 0
        FIX: use range(self.epoch + 1, n_epochs) to continue from the last epoch in case of resuming from checkpoint
        '''
        
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
            
            # Auto-save every epoch (for work on Colab)
            if True:
                self.save_checkpoint(f"backup_epoch_{epoch}.pt")

'''
change total stes based on dataset size and batch size:
for a dataset with 100k samples and batch size of 16, 
one epoch is 100000 / 16 = 6250 steps. For 10 epochs, total steps = 6250 * 10 = 62500
the warmup must be arround 5% to 10% of total steps, so 4000 steps is a good choice for warmup
'''
def get_scheduler(optimizer, d_model, warmup_steps=4000, total_steps=62500):
    """
    Scheduler with linear warmup and cosine decay (inspired by "Attention is All You Need" paper)
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    phase 1 (warmup): LR increases linearly from 0 to peak (d_model^(-0.5) * warmup_steps^(-0.5)) over warmup_steps
    phase 2 (decay): LR decreases following a cosine curve from the peak to near zero over the remaining steps (total_steps - warmup_steps)
    This scheduler is designed to work well with the Adam optimizer and the Transformer architecture
    """
    def lr_lambda(step):
        step = max(1, step)
        # linear warmup for the first warmup_steps, then cosine decay for the remaining steps

        if step < warmup_steps:
            # Warmup phase: LR grows linearly from 0 to peak
            return step / warmup_steps
        else:
            # Decay phase: cosine annealing, not 1/sqrt(step) too aggressive for later steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # LambdaLR applies the lr_lambda function to compute the learning rate at each step based on the global step count