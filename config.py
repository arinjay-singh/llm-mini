from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size = 1024  # context size
    vocab_size = 50257  # vocabulary size
    n_layer = 12  # number of transformer blocks
    n_head = 12  # number of attention heads
    n_embd = 768  # embedding dimension