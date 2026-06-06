"""
Base implementation of a standard transformer from the NRGPT paper
https://github.com/bhoov/nrgpt/tree/main
"""

# ================================== Imports ================================= #
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_config import TransformerConfig

# ======================== Base transformer components ======================= #


# --------------------------- Single Attention Head -------------------------- #
class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, config, masked=False):
        super().__init__()
        self.key = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.query = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embed, config.head_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # Set mask if desired
        if masked:
            self.register_buffer(
                "tril",
                torch.tril(
                    torch.ones(config.sequence_len, config.sequence_len), diagonal=0
                ),
            )

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        if hasattr(self, "tril"):
            # Apply mask to transformer
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        if hasattr(self, "tril"):
            # Apply mask to attention weights
            all_mask_rows = torch.all(
                self.tril[:T, :T] == 0, dim=-1, keepdim=True
            )  # (T, 1)
            wei = wei.masked_fill(all_mask_rows, 0.0)

        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


# -------------------------- Multi-Headed Attention -------------------------- #
class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, config: TransformerConfig, masked=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(config, masked=masked) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(config.head_size * config.n_head, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# --------------------------- Standard Linear Layer -------------------------- #
class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, config):
        super().__init__()
        h = config.ff_hid_factor * config.n_embed  # usually 4x n_embed
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, h),
            nn.ReLU(),
            nn.Linear(h, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------- Transformer block ---------------------------- #
class TransformerBlock(nn.Module):
    """Transformer block w/ parallel implementation"""

    def __init__(self, config):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = config.n_embed // config.n_head
        self.sa = MultiHeadAttention(config, head_size)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x, **kwargs):
        x = x + self.sa(self.ln1(x)) + self.ffwd(self.ln2(x))
        return x
