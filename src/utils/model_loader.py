"""
Model loading utilities for Weights & Biases, as well as other libraries.
"""

# ================================== Imports ================================= #
import os
import torch
import wandb

from energy_transformers.model_config import TransformerConfig
from energy_transformers.baseline_transformer import RecursiveCGPT
from energy_transformers.energy_transformer import RecursiveNRGPT


# ============================= W&B Model Loading ============================ #
def get_best_sweep_model(
    api, sweep_id, project_name, entity, wandb_config, model_class, device=None
):

    sweep = api.sweep(f"{entity}/{project_name}/{sweep_id}")
    finished_runs = [r for r in sweep.runs if r.state == "finished"]
    if not finished_runs:
        raise ValueError("No runs finished successfully. Cannot load model.")
    best_run = min(finished_runs, key=lambda r: r.summary.get("val_loss", float("inf")))
    print(
        f"Best Run ID: {best_run.id} | Val Loss: {best_run.summary.get('val_loss'):.4f}"
    )

    # 3. Download the best model's weights
    artifact_path = f"{entity}/{project_name}/model-{best_run.id}:latest"
    artifact = api.artifact(artifact_path)
    download_dir = artifact.download()

    # 4. Load the model for testing
    # Recreate the model config using the best run's saved parameters
    n_embed = wandb_config["parameters"]["n_embed"]["value"]
    n_layers = wandb_config["parameters"]["n_layers"]["value"]
    n_head = wandb_config["parameters"]["n_head"]["value"]
    sequence_len = wandb_config["parameters"]["sequence_len"]["value"]
    vocab_size = wandb_config["parameters"]["vocab_size"]["value"]
    n_classes = wandb_config["parameters"]["n_classes"]["value"]
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

    # Instantiate empty model and load weights
    model = model_class(model_config)
    unique_model_path = f"model_{best_run.id}.pth"
    weights_path = f"{download_dir}/{unique_model_path}"
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )

    # Move model to device if specified
    if device is not None:
        model = model.to(device)

    return model, best_run
