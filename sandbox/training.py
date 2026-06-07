"""
Functions to handle model training and logging.
"""

# ================================== Imports ================================= #
from dataclasses import dataclass

import numpy as np
import torch


# ====================== Helper configuration dataclass ====================== #
@dataclass
class TrainingConfig:
    num_epochs: int
    total_dataset_samples: int
    batch_size: int
    # Total batch rounds for the whole training, calculated post init
    total_train_iterations: int
    adamw_weight_decay: float
    scheduler_Tmax: int

    # LR Scheduling, defaults from nanochat
    lr_embedding: float = 0.3
    lr_unembedding: float = 8e-3
    lr_warmup_iters: int = 40
    lr_warmdown_ratio: float = 0.65
    lr_final_frac: float = 0.05

    # Post init auto calculation
    def __post_init__(self):
        if self.total_dataset_samples is not None and self.batch_size is not None:
            self.total_train_iterations = (
                self.total_dataset_samples * self.num_epochs
            ) // self.batch_size


# ======================== Main training loop function ======================= #
def train_epoch(model, dataloader, optimizer, device):
    # Enable training
    model.train()

    # Loss histories for logging
    loss_history = np.zeros(len(dataloader))
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y = y.squeeze()  # Remove extra dimension if present

        # Forward pass
        logits = model(X)
        loss = torch.nn.functional.cross_entropy(logits, y)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        # Save for stats
        loss_history[batch_idx] = loss.item()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return loss_history.mean(), loss_history.std()


# ======================= Auxiliary training functions ======================= #
def linear_learnrate_scheduler(it, config):
    """
    Trapezoidal learning rate scheduler, returns LR multiplier.

    Adapted from the nanochat repo
    """
    lr_final_frac = config.lr_final_frac
    warmdown_ratio = config.lr_warmdown_ratio
    warmup_iters = config.lr_warmup_iters
    total_iter = config.total_train_iterations
    warmdown_iters = int(total_iter * warmdown_ratio)

    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= total_iter - warmdown_iters:
        return 1.0
    else:
        progress = (total_iter - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * lr_final_frac
