"""
Common configuration class for energy transformer models.
"""

# ================================== Imports ================================= #
from dataclasses import dataclass

# ================= Config Class for all Transformer versions ================ #


@dataclass
class TransformerConfig:
    """Configuration class for energy transformer models"""

    sequence_len: int
    n_embed: int
    head_size: int
    n_head: int
    dropout: float
    ff_hid_factor: int
