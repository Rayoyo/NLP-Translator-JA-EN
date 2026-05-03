"""
model.py
--------
Transformer sequence-to-sequence model for Japanese–English translation,
built on top of PyTorch's :class:`torch.nn.Transformer`.

Architecture overview
~~~~~~~~~~~~~~~~~~~~~
* Shared or separate embeddings for source and target vocabularies.
* Sinusoidal positional encoding.
* Standard encoder–decoder Transformer (configurable depth & width).
* Linear projection head to target vocabulary logits.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Add sinusoidal positional encodings to token embeddings.

    Args:
        d_model: Embedding / model dimension.
        dropout: Dropout probability applied after adding the encoding.
        max_len: Maximum sequence length supported.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (1, max_len, d_model) – compatible with batch_first=True
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Tensor of the same shape with positional encodings added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """Encoder–decoder Transformer for sequence-to-sequence translation.

    Args:
        src_vocab_size: Source (Japanese) vocabulary size.
        tgt_vocab_size: Target (English) vocabulary size.
        d_model: Model / embedding dimension (default 256).
        nhead: Number of attention heads (default 8). Must divide *d_model*.
        num_encoder_layers: Encoder depth (default 3).
        num_decoder_layers: Decoder depth (default 3).
        dim_feedforward: Hidden size of the feed-forward sub-layer (default 512).
        dropout: Dropout probability (default 0.1).
        max_len: Maximum sequence length (default 512).
        pad_id: Padding token ID used to build key-padding masks (default 0).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for embedding and projection layers."""
        nn.init.xavier_uniform_(self.src_embedding.weight)
        nn.init.xavier_uniform_(self.tgt_embedding.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def make_src_key_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask where ``True`` marks padding positions.

        Args:
            src: Token-ID tensor of shape ``(batch, src_len)``.

        Returns:
            Bool tensor of shape ``(batch, src_len)``.
        """
        return src == self.pad_id

    def make_tgt_key_padding_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask where ``True`` marks padding positions.

        Args:
            tgt: Token-ID tensor of shape ``(batch, tgt_len)``.

        Returns:
            Bool tensor of shape ``(batch, tgt_len)``.
        """
        return tgt == self.pad_id

    def make_causal_mask(self, tgt_len: int, device: torch.device) -> torch.Tensor:
        """Return an upper-triangular causal mask for autoregressive decoding.

        Args:
            tgt_len: Length of the target sequence.
            device: Target device.

        Returns:
            Float tensor of shape ``(tgt_len, tgt_len)`` with ``-inf`` above
            the diagonal and ``0`` on and below.
        """
        return nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass (training / teacher-forcing).

        Args:
            src: Source token IDs, shape ``(batch, src_len)``.
            tgt: Target token IDs, shape ``(batch, tgt_len)``.
                 During training, this is the right-shifted ground truth.

        Returns:
            Logits of shape ``(batch, tgt_len, tgt_vocab_size)``.
        """
        src_pad_mask = self.make_src_key_padding_mask(src)
        tgt_pad_mask = self.make_tgt_key_padding_mask(tgt)
        tgt_causal_mask = self.make_causal_mask(tgt.size(1), src.device)

        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_causal_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.fc_out(out)

    @torch.no_grad()
    def translate(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 100,
    ) -> list:
        """Greedy decode a single source sentence.

        Args:
            src: Token-ID tensor of shape ``(1, src_len)`` (single sentence).
            bos_id: ID of the beginning-of-sequence token.
            eos_id: ID of the end-of-sequence token.
            max_len: Maximum number of tokens to generate.

        Returns:
            List of integer token IDs (excluding BOS, including EOS if generated).
        """
        self.eval()
        device = src.device

        src_pad_mask = self.make_src_key_padding_mask(src)
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        tgt_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_emb = self.pos_encoding(
                self.tgt_embedding(tgt_ids) * math.sqrt(self.d_model)
            )
            tgt_causal_mask = self.make_causal_mask(tgt_ids.size(1), device)
            dec_out = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            logits = self.fc_out(dec_out[:, -1, :])
            next_id = logits.argmax(dim=-1).item()
            tgt_ids = torch.cat(
                [tgt_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )
            if next_id == eos_id:
                break

        # Return token IDs without the leading BOS
        return tgt_ids[0, 1:].tolist()
