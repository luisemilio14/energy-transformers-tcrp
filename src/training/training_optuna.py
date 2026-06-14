"""
Functions to handle model training and logging.
"""

# ================================== Imports ================================= #
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
import optuna

from energy_transformers.model_config import TransformerConfig
from evaluation.evaluate import evaluate_cross_entropy, evaluate_acc
from training.training_config import TrainingConfig


# ======================== Main training loop function ======================= #
def train_epoch(model, dataloader, optimizer, lr_scheduler, config: TrainingConfig):
    # Enable training
    model.train()

    # Loss histories for logging
    avg_loss = 0.0
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(config.device), y.to(config.device)
        y = y.squeeze()  # Remove extra dimension if present

        # Forward pass
        logits = model(X)
        loss = torch.nn.functional.cross_entropy(logits, y)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.clip_grad_norm
        )
        optimizer.step()
        lr_scheduler.step()

        # Save for stats
        avg_loss += loss.item()

        if batch_idx % config.print_batch_interval == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return avg_loss / len(dataloader)


# ======================== Hyperparameter optimization ======================= #
def hyperparameter_optimization(
    trial,
    model_class,
    train_data: DataLoader,
    val_data: DataLoader,
    train_params: dict,
    model_params: dict,
    predef_train_config: TrainingConfig | None = None,
    predef_model_config: TransformerConfig | None = None,
):
    # --- Training Hyperparameters --- #
    if predef_train_config is None:
        # Optimize learning parameters
        # TODO: handle other parameters, make it more robust for scalars and ranges
        lr = trial.suggest_float("lr", *train_params["lr"])
        lr_final_frac = trial.suggest_float(
            "lr_final_frac", *train_params["lr_final_frac"]
        )
        weight_decay = trial.suggest_float(
            "weight_decay", *train_params["weight_decay"]
        )
        clip_grad_norm = trial.suggest_float(
            "clip_grad_norm", *train_params["clip_grad_norm"]
        )

        train_cfg = TrainingConfig(
            train_data=train_data,
            val_data=val_data,
            num_epochs=train_params["n_epochs"],
            total_dataset_samples=train_params["total_dataset_samples"],
            batch_size=train_params["batch_size"],
            device=train_params["device"],
            lr_warmup_iters=train_params["lr_warmup_iters"],
            lr_warmdown_ratio=train_params["lr_warmdown_ratio"],
            # Optimized hyperparameters
            lr=lr,
            lr_final_frac=lr_final_frac,
            weight_decay=weight_decay,
            clip_grad_norm=clip_grad_norm,
        )
    else:
        train_cfg = predef_train_config

    # --- Model Hyperparameters --- #
    if predef_model_config is None:
        # Search over model hyperparameters
        # Suggest categorical bc we'll be picking one of a few discrete options
        n_embed = trial.suggest_categorical("n_embed", model_params["n_embed"])
        n_layers = trial.suggest_categorical("n_layers", model_params["n_layers"])
        n_head = trial.suggest_categorical("n_heads", model_params["n_heads"])

        # --- Generate model --- #
        model_config = TransformerConfig(
            # Hyperparameters to optimize
            n_embed=n_embed,
            n_layers=n_layers,
            n_head=n_head,
            # Heuristics
            head_size=n_embed // n_head,
            dropout=0.2,  # Fixed dropout for now
            ff_hid_factor=4,  # Fixed feedforward hidden size factor for now
            # Fixed, data-determined parameters
            masked_attention=True,
            sequence_len=model_params["sequence_len"],
            vocab_size=model_params["vocab_size"],
            n_classes=model_params["n_classes"],
        )
    else:
        model_config = predef_model_config

    # --- Generate model --- #
    model = model_class(model_config).to(train_params["device"])
    model_size = sum(p.numel() for p in model.parameters())
    trial.set_user_attr("model_size", model_size)

    # --- Optimizer and LR scheduler --- #
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda it: linear_learnrate_scheduler(it, train_cfg)
    )

    # --- Training loop --- #
    for ep in np.arange(train_cfg.num_epochs):
        # Train the model for 1 epoch
        avg_epoch_loss = train_epoch(
            model, train_cfg.train_data, optimizer, lr_scheduler, train_cfg
        )

        # Evaluate after every epoch
        # Since datasets are small, we really dont need intra-epoch evaluation
        val_acc = evaluate_acc(model, train_cfg.val_data, train_cfg.device)
        val_loss = evaluate_cross_entropy(model, train_cfg.val_data, train_cfg.device)
        # TODO: register values on trial for logging

        # Report intermediate results to Optuna for pruning
        trial.report(val_loss, ep)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # At end of trial, return final validation loss for optimization
    return val_loss


# ======================= Auxiliary training functions ======================= #
def linear_learnrate_scheduler(it, config: TrainingConfig):
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
