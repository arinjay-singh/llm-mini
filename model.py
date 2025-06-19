# GPT-style LLM Architecture

import torch
import torch.nn as nn
from torch.nn import functional as F

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
    def __init__(self, n_embd: int, n_hidden: int):
        super().__init__()
        
        self.fc1 = nn.Linear(n_embd, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_embd)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        
        return x
    
