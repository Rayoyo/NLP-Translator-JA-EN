"""
Transformer from scratch for translation EN <-> JP
Implements: Embeddings + Positional Encoding, Multi-Head Attention, Encoder/Decoder layers, Masking
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)  # droput layer to prevent overfitting (to reset 10% of the neurons)
        
        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        '''
        torch.arange(0, 5000) → tensor([0, 1, 2, ..., 4999])
        .unsqueeze(1) → tensor([[0], [1], [2], ..., [4999]])
        used for broadcasting in subsequent operations
        '''

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        '''
        torch.arange(0, d_model, 2) → [0, 2, 4, ..., 510] (if d_model=512)
        div_term = exp([0, 2, 4, ..., 510] * (-log(10000)/512))
        torch.exp() → applies the exponential function to each element, resulting in a tensor of shape (d_model/2,)
        Computes the denominator for the positional encoding formula
        '''
        
        pe[:, 0::2] = torch.sin(position * div_term)
        '''
        pe[:, 0::2] → selects all even indices in the last dimension (0, 2, 4, ...)
        position * div_term → computes the argument for the sine function for even dimensions'''

        pe[:, 1::2] = torch.cos(position * div_term)
        # Odd columns (1, 3, 5, ...) → cosine function -- sin/cos alternate for even/odd dimensions
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        '''
        .unsqueeze(0) → adds batch dimension: (1, 5000, 512)
        .transpose(0, 1) → swap dim 0 and 1: (5000, 1, 512)
        for compatibility with the input shape (seq_len, batch_size, d_model)
        '''
        
        self.register_buffer('pe', pe)
        '''
        Registers the positional encodings as a buffer, so they are saved with the model state -- not trainable
        Saved with the model, not updated by the gradient descent
        '''

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :] 
        '''
        self.pe[:x.size(0), :] → takes only the first seq_len positions
        Broadcasting: pe (seq_len, 1, d_model) + x (seq_len, batch, d_model)
        "1" expands to batch size, so each position encoding is added to all examples in the batch
        '''
        return self.dropout(x)  # Apply dropout (randomly zeros items) and return the result


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divided for n_heads"
        
        self.d_model = d_model  # dim of the vector of each word (ex 512)
        self.n_heads = n_heads  # number of attention heads (ex 8) -> for different perspectives of the input phrase
        self.d_k = d_model // n_heads # dim of each head (ex 512/8 = 64)
        
        self.W_q = nn.Linear(d_model, d_model)  # query - what we are searching for
        self.W_k = nn.Linear(d_model, d_model)  # key - what offers every word
        self.W_v = nn.Linear(d_model, d_model)  # value - what is the content
        self.W_o = nn.Linear(d_model, d_model)  # final projection - combines the heads back into a single vector
        
        self.dropout = nn.Dropout(dropout)  # dropout on attention weights to prevent overfitting
        self.scale = math.sqrt(self.d_k)    # scaling factor to prevent large dot product values (leading to small gradients)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V: (batch, n_heads, seq_len, d_k)
        # ex    Q: (   32,      8,     20,    64) -> batch of 32, 8 heads, sequence length 20, head dimension 64
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, heads, seq, seq)
        '''
        swap the last 2 dimensions of K to compute the dot product between Q and K
        K: (batch, heads, seq, d_k) → K.T: (batch, heads, d_k, seq)
        matmul: (batch, heads, seq, d_k) @ (batch, heads, d_k, seq) = (batch, heads, seq, seq)
        result: matrix seq x seq where each element (i, j) represents the attention score between position i and j for that head
        / self.scale → scaling to prevent large values that can lead to small gradients (softmax saturation)
        '''
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            '''
            mask == 0 → positions to mask (False/0) 
            masked_fill → sets those positions to a very large negative value (-1e9) 
            -1e4 in softmax → exp(-10000) ≈ 0, that position will be oscured
            '''
        
        attn = torch.softmax(scores, dim=-1)
        '''
        softmax over the last dimension (seq_len) to get attention weights that sum to 1 for each query position
        Normalize each row: sum of attentions = 1 for each query position

        '''
        attn = self.dropout(attn)    # Randomly resets some attention connections
        out = torch.matmul(attn, V)  # (batch, heads, seq, d_k)
        ''''
        (batch, heads, seq, seq) @ (batch, heads, seq, d_k) = (batch, heads, seq, d_k)
        For each word, it adds the Values ​​of the words it paid attention to, weighted by attn
        '''

        return out, attn
        ''''
        out: resulting tensor after applying attention weights to the values
        attn: attention matrix (useful for visualization)
        '''
    
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)  # initial query shape: (batch_size, seq_len, d_model)
        
        ''' Linear projections + split into heads
        view: resizes the tensor. -1 = "automatically calculate"
        transpose: swaps dimensions 1 and 2 to group by head
        '''# (batch, seq, d_model) -> (batch, seq, heads, d_k) -> (batch, heads, seq, d_k)'''
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        '''
        Step 1: W_q(query) → (batch, seq, d_model)
        Step 2: .view(batch, seq, 8, 64) → split d_model into 8 64-bit heads
        Step 3: .transpose(1, 2) → (batch, 8, seq, 64)
        Now each "head" has its own sequence of queries
        '''
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        x, attn = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        '''
        .transpose(1, 2): (batch, heads, seq, d_k) → (batch, seq, heads, d_k)
        .contiguous() → ensures the tensor is stored contiguously in memory 
        .view(batch, seq, d_model): recombine the heads 8x64 = 512
        '''
        
        return self.W_o(x) # final projection (batch, seq, d_model) to d_model


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # d_ff = 4 × d_model standard of the model: 512 → 2048 (expansion)
        self.linear2 = nn.Linear(d_ff, d_model) # second projection 2048 → 512 (compression)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() # add non linearity → 0, max
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
        # Flow:x → linear1 (512→2048) → ReLU → Dropout → linear2 (2048→512) -- same input dimension
        
    


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout) # each word looks at ALL the other words in the input sentence
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # LayerNorm: normalizes the values ​​for each vector (avg=0, std=1) avoiding too high values during training
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, src_mask=None):
        # Self-attention with residual connection
        attn_out = self.self_attn(x, x, x, src_mask)
        # Query=x, Key=x, Value=x: "self" attention, src_mask: to mask padding

        x = self.norm1(x + self.dropout1(attn_out))
        '''
        x + dropout(attn_out): residual connection (skip attention)
        norm1: normalizes the output of the attention + residual to stabilize training
        the skip allows to avoid vanishing gradients
        '''
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        # same pattern: feed-forward + residual + normalization
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # mask to prevent attending to future tokens
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # cross-attention to encoder output (queries from decoder, keys/values from encoder)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked self-attention (causal)
        attn_out = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # Cross-attention with encoder output
        attn_out = self.cross_attn(x, enc_output, enc_output, src_mask)
        '''
        Query = x (decoder), Key = enc_output, Value = enc_output
        decoder attends to encoder output to gather relevant information from the input sentence
        '''
        x = self.norm2(x + self.dropout2(attn_out))
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))
        
        return x


class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size,       # dimension of the source vocabulary (number of unique tokens)
        tgt_vocab_size,       # dimension of the target vocabulary (number of unique tokens)
        d_model=512,          # dimension of the embedding vectors (and model hidden states)
        n_heads=8,            # number of attention heads (8)
        n_encoder_layers=6,   # number of encoder layers (6)
        n_decoder_layers=6,   # number of decoder layers (6)
        d_ff=2048,            # dimension of the feed-forward layer (2048, typically 4x d_model)
        max_len=5000,         # maximum sequence length
        dropout=0.1,          # dropout rate for regularization
        pad_idx=0             # index of the padding token in the vocabulary
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embeddings matrix vocab_size x d_model
        # Each token in the vocabulary is mapped to a d_model-dimensional vector (512)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # nn.Embedding: lookup table. Input: index (batch_size, src_len), Output: vector (batch_size, src_len, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        '''
        Scaling factor: embeddings are multiplied by √d_model
        to balance the magnitude with respect to the positional encoding
        '''
        
        # Scale embeddings by sqrt(d_model)
        self.scale = math.sqrt(d_model)
        
        # Encoder stack - 6 layers of self-attention + feed-forward
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder stack - 6 layers of self-attention + feed-forward
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # Output projection: maps decoder output to target vocabulary size for prediction
        # (batch_size, tgt_len, tgt_vocab_size) - for each position in the target sequence, predicts a distribution
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        
        # Initialize parameters with Xavier uniform (good for deep networks)
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                '''
                Xavier: Initialized weights based on fan_in/fan_out to maintain variance across layers
                Helps to prevent vanishing/exploding gradients in deep networks
                Not applied to bias (dim=1) or batch norm
                '''
    
    def make_src_mask(self, src):
        # src: (batch_size, src_len) → integer ID tensor
        # mask: (batch_size, 1, 1, src_len)
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        '''
        src != 0: Boolean tensor (True where there is a real word, False where <PAD>)
        .unsqueeze(1): add dimension → (batch, 1, src_len)
        .unsqueeze(2): add dimension → (batch, 1, 1, src_len)
        final shape: (batch, 1, 1, src_len)
        "1" expands in broadcasting with (batch, heads, seq, seq)
        '''

        # Attention: where mask=False, the score becomes -1e4 (ignored)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        # tgt: (batch_size, tgt_len)
        tgt_len = tgt.size(1)
        
        # Padding mask: (batch_size, 1, 1, tgt_len)
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Causal mask (look-ahead): (1, 1, tgt_len, tgt_len) to mask future tokens
        causal_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device)
        ).bool().unsqueeze(0).unsqueeze(0) 
        # lower triangular matrix:  attending to current and previous tokens & .bool() to convert to boolean mask
        
        # Combine: padding AND causal -- mask both padding and future tokens
        tgt_mask = tgt_pad_mask & causal_mask
        return tgt_mask
    
    def encode(self, src, src_mask=None):
        # src: (batch_size, src_len)
        x = self.src_embedding(src) * self.scale 
        # every id is mapped to a vector and scaled by sqrt(d_model) to balance with positional encoding
        
        
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)  # (batch, seq, d_model)
        '''
        .transpose(0, 1): (batch, seq, 512) → (seq, batch, 512)
        pos_encoding expects (seq, batch, d_model) and adds positional encodings
        .transpose(0, 1) → back to (batch, seq, d_model)
        '''
        
        for layer in self.encoder_layers: # pass through each of the 6 encoder layers
            x = layer(x, src_mask)
            
        return x # (batch_size, src_len, d_model) - contextualized representations of the input sequence
    
    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        # tgt: (batch_size, tgt_len) - already generated tokens (input to the decoder)
        # enc_output: (batch_size, src_len, d_model) - output from the encoder to attend to
        x = self.tgt_embedding(tgt) * self.scale  # target embedding, same logic as source embedding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        for layer in self.decoder_layers:  # pass through each of the 6 decoder layers
            x = layer(x, enc_output, src_mask, tgt_mask)
            
        return x
    
    def forward(self, src, tgt): 
        '''
        called during training
        src: source sequence (batch_size, src_len)
        tgt: target sequence (batch_size, tgt_len) - with teacher forcing (includes BOS and EOS)
        '''
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        output = self.fc_out(dec_output)
        '''
        (batch, tgt_len, 512) → (batch, tgt_len, vocab_size)
        For each position in the target sequence, we get a distribution over the target vocabulary
        '''
        return output
    
    def translate(self, src, max_len=50, bos_id=1, eos_id=2):
        """
        Inference: greedy decoding
        translate(): called during inference (sentence generation)
        src: (batch_size, src_len) or (src_len,)
        """
        self.eval()                 # deactivate dropout for inference, batch norm uses running stats instead of batch stats
        with torch.no_grad():       # no gradient calculation for efficiency
            if src.dim() == 1:
                src = src.unsqueeze(0)
                # If single vector (seq_len,), add batch dimension → (1, seq_len)
            
            src_mask = self.make_src_mask(src)
            enc_output = self.encode(src, src_mask)
            # code source sentence once
            
            # Start with BOS token - beginning of sentence
            tgt = torch.ones(src.size(0), 1).fill_(bos_id).long().to(src.device) #(batch_size, 1) filled with <BOS>
            
            for _ in range(max_len):
                tgt_mask = self.make_tgt_mask(tgt)    # casual masking: can see only current and previous tokens
                dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
                output = self.fc_out(dec_output) # (batch_size, tgt_len, vocab_size) - get distribution for the current target sequence
                
                # Get next token (greedy)
                next_token = output[:, -1:].argmax(dim=-1)   # only just generated token
                tgt = torch.cat([tgt, next_token], dim=1)    # append to target sequence for next iteration (batch_size, tgt_len+1)
                
                # Stop if all sequences generated EOS
                if (next_token == eos_id).all():
                    break
            
            return tgt
            # output: (batch_size, generated_seq_len) - sequence of token IDs including BOS and EOS


def count_parameters(model):
    '''
    model.parameters() returns an iterator over all the parameters of the model
    p.numel(): number of elements in the tensor (eg, a weight matrix of shape (512, 512) has 262144 parameters)
    if p.requires_grad: only count trainable parameters (exclude buffers like pe)
    sum everything ~93 million parameters for the full transformer (6 encoder + 6 decoder layers) 
        with d_model=512, n_heads=8, d_ff=2048
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)