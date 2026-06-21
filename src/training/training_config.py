"""
General config object for training runs
"""

# ================================== Imports ================================= #
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader


# =========================== TrainingConfig class =========================== #
@dataclass
class TrainingConfig:
    """Configuration for training runs, including data buffers and hyperparameters."""

    # Buffers for data
    train_data: DataLoader
    val_data: DataLoader

    # Training paramters
    num_epochs: int
    total_dataset_samples: int
    batch_size: int
    device: torch.device
    total_train_iterations: int = 0  # Calculated post init

    # Training Hyperparameters
    lr: float = 3e-4
    lr_warmup_iters: int = 10
    lr_warmdown_ratio: float = 0.65
    lr_final_frac: float = 0.05
    weight_decay: float = 1e-2
    clip_grad_norm: float = 1.0

    # Verbose options
    print_batch_interval: int = 100

    # Post init auto calculation
    def __post_init__(self):
        if self.total_dataset_samples is not None and self.batch_size is not None:
            if self.batch_size == 0:
                # If batch size is zero, treat as full-batch training
                self.total_train_iterations = self.num_epochs
            else:
                # Total iter = dataset size * epochs / batch size
                self.total_train_iterations = (
                    self.total_dataset_samples * self.num_epochs
                ) // self.batch_size
