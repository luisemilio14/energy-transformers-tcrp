"""
Functions to handle model training and logging.
"""

# ================================== Imports ================================= #
from dataclasses import dataclass

import numpy as np
from optuna import trial
import torch
from torch.utils.data import DataLoader
import wandb

from energy_transformers.model_config import TransformerConfig
from evaluation.evaluate import evaluate_cross_entropy, evaluate_acc
from training.training_config import TrainingConfig


# ======================== Main training loop function ======================= #
def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainingConfig,
) -> float:
    """Train the model for one epoch and return the average loss.

    Args:
        model: The PyTorch model to train.
        dataloader: DataLoader providing the training data.
        optimizer: The optimizer to use for training.
        lr_scheduler: Learning rate scheduler to step after each batch.
        config: TrainingConfig containing training hyperparameters and settings.
    """
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

        # Log batch loss
        wandb.log({"batch_loss": loss.item()})

        # Save for stats
        avg_loss += loss.item()

        if batch_idx % config.print_batch_interval == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return avg_loss / len(dataloader)


# ======================== Hyperparameter optimization ======================= #
def train(
    model_class,
    config=None,
    device: torch.device | None = None,
    train_data: DataLoader = DataLoader([]),
    val_data: DataLoader = DataLoader([]),
    predef_train_config: TrainingConfig | None = None,
    predef_model_config: TransformerConfig | None = None,
):
    with wandb.init(config=config):
        config = wandb.config

        # Post-init config processing
        total_dataset_samples = len(train_data.dataset)

        # --- Training Hyperparameters --- #
        if predef_train_config is None:
            # Optimize learning parameters
            lr = config.lr
            lr_final_frac = config.lr_final_frac
            weight_decay = config.weight_decay
            clip_grad_norm = config.clip_grad_norm
            num_epochs = config.n_epochs
            batch_size = config.batch_size
            total_dataset_samples = total_dataset_samples
            lr_warmup_iters = config.lr_warmup_iters
            lr_warmdown_ratio = config.lr_warmdown_ratio

            train_cfg = TrainingConfig(
                train_data=train_data,
                val_data=val_data,
                num_epochs=num_epochs,
                total_dataset_samples=total_dataset_samples,
                batch_size=batch_size,
                device=device,
                lr_warmup_iters=lr_warmup_iters,
                lr_warmdown_ratio=lr_warmdown_ratio,
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
            n_embed = config.n_embed
            n_layers = config.n_layers
            n_head = config.n_head
            sequence_len = config.sequence_len
            vocab_size = config.vocab_size
            n_classes = config.n_classes

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
                sequence_len=sequence_len,
                vocab_size=vocab_size,
                n_classes=n_classes,
            )
        else:
            model_config = predef_model_config

        # --- Generate model --- #
        model = model_class(model_config).to(device)
        # model_size = sum(p.numel() for p in model.parameters())
        # TODO: log model and training config objects as artifacts

        # --- Optimizer and LR scheduler --- #
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda it: linear_learnrate_scheduler(it, train_cfg)
        )
        # TODO: register optimizer and lr scheduler artifacts

        # --- Training loop --- #
        best_val_acc = 0.0
        best_model_state = None
        for ep in np.arange(train_cfg.num_epochs):
            # Train the model for 1 epoch
            avg_epoch_loss = train_epoch(
                model, train_cfg.train_data, optimizer, lr_scheduler, train_cfg
            )

            # Evaluate after every epoch
            # Since datasets are small, we really dont need intra-epoch evaluation
            val_acc = evaluate_acc(model, train_cfg.val_data, train_cfg.device)
            val_loss = evaluate_cross_entropy(
                model, train_cfg.val_data, train_cfg.device
            )

            # Log results
            wandb.log(
                {
                    "epoch": ep,
                    "train_loss": avg_epoch_loss,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                }
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()

        # Save model, optimizer and lr scheduler artifacts
        artifact = wandb.Artifact(
            f"model-{wandb.run.id}",
            type="model",
            description=f"{model_class.__name__} for the experiment where multiple model sizes are tested. See metadata for hyperparameters.",
            metadata={
                "model_class": model_class.__name__,
                "model_config": model_config.__dict__,
                "train_config": train_cfg.__dict__,
            },
        )
        torch.save(best_model_state, "model.pth")
        artifact.add_file("model.pth")
        wandb.log_artifact(artifact)


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
