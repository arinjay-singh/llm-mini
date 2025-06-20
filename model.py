# GPT-style LLM Architecture

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import GPTConfig
import inspect

# Multi-head self-attention mechanism
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        
        self.n_head = n_head
        self.n_embd = n_embd
        
        # ensure that the embedding dimension is divisible by the number of heads
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"
        
        self.head_dim = n_embd // n_head
        
        # linear layers for query, key, and value projections
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.out = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        # B: batch size, T: sequence length (time steps), C: embedding dimension (channel size)
        B, T, C = x.size() 
        
        # project inputs to queries, keys, and values
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        
        # reshape for multi-head attention
        queries = queries.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
        keys = keys.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
        values = values.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
        
        # perform scaled dot-product attention (flash attention)
        y = F.scaled_dot_product_attention(queries, keys, values, is_causal=True) # (B, n_head, T, head_dim)
        
        # reshape attention output back to (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # apply final linear layer
        y = self.out(y)
        
        return y
        
# Multi-layer Perceptron (MLP) block
class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        
        n_hidden = 4 * n_embd  # 4 times the embedding dimension
        
        self.fc1 = nn.Linear(n_embd, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_embd)
        self.act = nn.GELU(approximate='tanh')
        
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        
        return x
    
# Transformer Block
class Transformer(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_embd, n_head)
        self.dropout1 = nn.Dropout(0.1)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.dropout2 = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor):
        # first layer normalization and self-attention with residual connection
        x = x + self.dropout1(self.attn(self.ln1(x)))
        
        # second layer normalization and MLP with residual connection
        x = x + self.dropout2(self.mlp(self.ln2(x)))
        
        return x
    
    
# GPT-style model
class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        vocab_size = config.vocab_size
        n_embd = config.n_embd
        n_head = config.n_head
        n_layers = config.n_layer
        
        self.model = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),  # token embeddings
                wpe=nn.Embedding(1024, n_embd),  # positional embeddings
                dropout=nn.Dropout(0.1), # dropout layer
                h=nn.ModuleList([Transformer(n_embd, n_head) for _ in range(n_layers)]),  # stack of transformer blocks
                ln_f=nn.LayerNorm(n_embd),  # final layer normalization
                lm_head=nn.Linear(n_embd, vocab_size, bias=False)  # output linear layer for language modeling
            )
        )
        
        # weight tying with embedding matrix and output projection layer
        self.model.lm_head.weight = self.model.wte.weight
        
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        # idx are token indices, targets are optional labels for training
        
        B, T = idx.size()  # B: batch size, T: sequence length
        
        assert T <= 1024, "Sequence length exceeds maximum of 1024 tokens"
        
        # get token and positional embeddings
        pos = torch.arange(0, T, device=idx.device) # (T,)
        pos_emb = self.model.wpe(pos)  # (T, n_embd)
        tok_emb = self.model.wte(idx)  # (B, T, n_embd)
        
        # combine token and positional embeddings and apply dropout
        x = self.model.dropout(tok_emb + pos_emb)  # (B, T, n_embd)
        
        # pass through transformer blocks
        for block in self.model.h:
            x = block(x)
            
        # apply final layer normalization
        x = self.model.ln_f(x)  # (B, T, n_emb
        
        # compute logits
        logits = self.model.lm_head(x)  # (B, T, vocab_size)
        
        # compute loss if targets are provided
        if targets is not None:
            # compute loss using cross-entropy
            # logits reshaped to (B*T, vocab_size), targets reshaped to (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) 
            
        return logits, loss
    
    def setup_optimizer(self, weight_decay, learning_rate, device):
        # start with all candidate parameters (that require gradients)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups
        # any parameters that are 2D will be decayed
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms do not
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(
            f"num of decayed parameter tensors: {len(decay_params)}, with {num_decay_params} total parameters"
        )
        print(
            f"num of non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params} total parameters"
        )
        # create AdamW optimizer an duse the fused version if possible
        fused_available = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )  # confirm if fused is available (kernel fusion for AdamW update)
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer
    
    