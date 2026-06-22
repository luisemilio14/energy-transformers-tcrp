"""
Implements Energy Transformer architecture, based on the NRGPT paper
"""

# ================================== Imports ================================= #
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad

from .model_config import TransformerConfig
from .baseline_transformer import RecursiveCGPT


# ======================= Aux LayerNorm implementation ======================= #
class BareLayerNorm(nn.Module):
    """
    LayerNorm but without learnable weights, only bias

    The weights are omitted as a learning parameter since they are absorbed by
    the other learned parameters of the network, see Section 2.3 of the NRGPT
    paper.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        # Keep the learnable bias
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

        # REMOVE the frozen weight parameter completely.
        # It causes AMP CUDA kernel crashes.

    def forward(self, x):
        return F.layer_norm(
            x,
            normalized_shape=self.normalized_shape,
            eps=self.eps,
            bias=self.bias,
            weight=None,  # Explicitly pass None here
        )


# =============== Energy-based attention and feedforward layers ============== #


# --------------------------- Single Attention head -------------------------- #
class EnergyHead(nn.Module):
    """one head of energy-based self-attention"""

    def __init__(self, config):
        super().__init__()
        # Jh = W^K' @ W^Q in the paper
        self.Jh = nn.Linear(config.n_embed, config.n_embed, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        if config.masked_attention:
            self.register_buffer(
                "tril",
                torch.tril(
                    torch.ones(config.sequence_len, config.sequence_len), diagonal=0
                ),
            )

    def forward(self, x):
        # input of size (batcabsorbed by h, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        xJh = self.Jh(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = x @ xJh.transpose(-2, -1)  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        if hasattr(self, "tril"):
            # Apply mask to transformer
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        if hasattr(self, "tril"):
            # Apply mask if defined
            all_mask_rows = torch.all(
                self.tril[:T, :T] == 0, dim=-1, keepdim=True
            )  # (T, 1)
            wei = wei.masked_fill(all_mask_rows, 0.0)

        wei = self.dropout(wei)
        out = wei @ xJh  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return -out  # this is grad of energy

    def energy(self, x):
        B, T, C = x.shape
        xJh = self.Jh(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = x @ xJh.transpose(-2, -1)  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        if hasattr(self, "tril"):
            # Apply mask to transformer
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        logsumexp_wei = torch.logsumexp(wei, dim=-1)  # (B, T)

        if hasattr(self, "tril"):
            # Apply mask if defined
            all_mask_rows = torch.all(
                self.tril[:T, :T] == 0, dim=-1, keepdim=True
            )  # (T, 1)
            logsumexp_wei = logsumexp_wei.masked_fill(all_mask_rows.squeeze(-1), 0.0)

        return -logsumexp_wei  # (B, T)


# -------------------------- Multi-headed attention -------------------------- #
class MultiHeadEnergyAttention(nn.Module):
    """multiple heads of energy-based self-attention in parallel"""

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([EnergyHead(config) for _ in range(config.n_head)])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = 0.0
        for h in self.heads:
            out += h(x)
        out = self.dropout(out)
        return out

    def energy(self, x):
        E = 0.0
        for h in self.heads:
            E += h.energy(x)
        return E


# ============================ Full energy models ============================ #


# -- Base class for implementations that directly use the grad for fwd pass -- #
class GradENet(nn.Module):
    """
    Generic class for energy-based networks.

    Energy-based possess two characteristics:
    (1) A scalar energy function E(x)
    (2) A forward pass defined as the gradient of the energy, such that
        the output guarantees that the energy is monotonically decreasing.
    """

    def __init__(self, config):
        super().__init__()
        self.net = self.define_network(config)
        self.gf = grad(self.energy)

    def define_network(self, config):
        raise NotImplementedError("This method should be implemented by subclasses")

    def energy(self, x):
        return -(self.net(x) ** 2).sum()

    def forward(self, x):
        return self.gf(x)


# --------- Gradient-based feedforward layer using energy formulation -------- #
class GradFeedForward(GradENet):
    """
    Implements an energy based, virtual two layer feedforward network, where the forward pass is defined as the

    Forward pass defined as the gradient of the L2 norm of a linear+GELU net,
    such that 𝛁E = 2W(GELU(Wx+b) * GELU'(Wx+b)).
    """

    def __init__(self, config):
        super().__init__(config)

    def define_network(self, config):
        h = config.ff_hid_factor * config.n_embed
        return nn.Sequential(nn.Linear(config.n_embed, h), nn.GELU())


# ---------------------- Energy-based transformer block ---------------------- #
class EnergyTransformerBlock(nn.Module):
    """Energy Transformer block with GradFeedForward"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = MultiHeadEnergyAttention(config)
        self.ffwd = GradFeedForward(config)
        # BareLayerNorm is problematic for AMP CUDA kernels, so we use regular LayerNorm here
        # self.ln = BareLayerNorm(config.n_embed)
        self.ln = nn.LayerNorm(config.n_embed)
        self.proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.scale_ff = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x, **kwargs):
        # Parallel attention: x^{t+1} = x^t + AT(g) + FF(g)
        x = x - self.proj(self.attn(self.ln(x)) + self.scale_ff * self.ffwd(self.ln(x)))
        return x


# -------------- Energy-based version of the Recursive GPT class ------------- #s
class RecursiveNRGPT(RecursiveCGPT):
    def __init__(self, config: TransformerConfig):
        super().__init__(config, block_class=EnergyTransformerBlock)
