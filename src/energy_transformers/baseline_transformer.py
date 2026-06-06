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

    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.query = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.is_masked = config.masked_attention
        self.dropout = nn.Dropout(config.dropout)

        # Set mask if desired
        if self.is_masked:
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

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
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
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x, **kwargs):
        x = x + self.sa(self.ln1(x)) + self.ffwd(self.ln2(x))
        return x


# -- 'GPT' Net - Transformer + Embedd + Linear, modified for classification -- #
class ClassifciationGPT(nn.Module):
    def __init__(self, config: TransformerConfig, block_class=TransformerBlock):
        super().__init__()
        self.config = config
        self.n_layers = config.n_layers
        self.block_class = block_class

        # Configure embeddings and positional encoding
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_emb = nn.Embedding(config.sequence_len, config.n_embed)
        self.register_buffer("pos_idx", torch.arange(config.sequence_len))

        # Build chained transformer blocks
        self.blocks = self.get_blocks(config)

        # Final layernorm + classification layer
        # * Why not the BareLayerNorm here?
        self.norm_f = nn.LayerNorm(config.n_embed)
        self.lin_f = nn.Linear(config.n_embed, config.n_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def block_forward(self, x):
        return self.blocks(x)

    def get_blocks(self, config):
        return nn.Sequential(
            *[self.block_class(config) for _ in range(config.n_layers)]
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        device = x.device  # For GPU

        # Pass throigh embeddings + posencode
        tkn_emb = self.token_emb(x)
        pos_emb = self.position_emb(self.pos_idx[:T])  # TODO understand
        g = tkn_emb + pos_emb

        # Pass through block
        g = self.block_forward(g)
        g = self.norm_f(g)
        logits = self.lin_f(g)

        # TODO: On debug, check shapes
        # Calc loss for gradient
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


# ------------------- Recursive Version of the GPT Version ------------------- #
class RecursiveCGPT(ClassifciationGPT):
    def __init__(self, config, block_class=TransformerBlock):
        super().__init__(config, block_class=block_class)

    def get_blocks(self, config):
        # Only a single block that will be applied recursively
        return self.block_class(config)

    def block_forward(self, x):
        B, T, C = x.shape
        for _ in range(self.n_layers):
            x = self.blocks(x)
        return x
