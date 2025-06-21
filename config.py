from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024  # context size
    vocab_size: int = 50257  # vocabulary size
    n_layer: int = 12  # number of transformer blocks
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension