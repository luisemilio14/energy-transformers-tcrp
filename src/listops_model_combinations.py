# ================================== Imports ================================= #
# Standard library imports
from argparse import ArgumentParser
from functools import partial
import os

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
import wandb
import torch
import yaml

# Custom imports
from data_handling.listops32 import generate_listops32_dataloader
from training.training_wandb import train
from evaluation.evaluate import evaluate_cross_entropy, evaluate_acc
from energy_transformers.baseline_transformer import RecursiveCGPT
from energy_transformers.energy_transformer import RecursiveNRGPT
from utils.model_loader import get_best_sweep_model


# ========================== Pipeline function call ========================== #
def train_listops32_model_combinations(config_path) -> None:
    # Setup env vars
    load_dotenv()
    entity = os.environ.get("WANDB_ENTITY")
    if entity is None:
        raise ValueError("WANDB_ENTITY not found in environment variables.")

    # Load configs
    config = yaml.safe_load(open(config_path, "r"))
    data_config = config.get("listops32", {})["data_processing"]
    wandb_path = config.get("all", {}).get("wandb_config_file", "wandb_config.yaml")
    wandb_config = yaml.safe_load(open(wandb_path, "r"))

    # Create dataloaders
    xtr = pd.read_parquet(data_config["tk_train_data_path"])
    ytr = pd.read_parquet(data_config["tk_train_label_path"])
    xval = pd.read_parquet(data_config["tk_val_data_path"])
    yval = pd.read_parquet(data_config["tk_val_label_path"])
    xte = pd.read_parquet(data_config["tk_test_data_path"])
    yte = pd.read_parquet(data_config["tk_test_label_path"])

    batch_size = wandb_config["parameters"]["batch_size"]["value"]
    train_dataloader = generate_listops32_dataloader(xtr, ytr, batch_size, shuffle=True)
    val_dataloader = generate_listops32_dataloader(
        xval, yval, batch_size, shuffle=False
    )
    test_dataloader = generate_listops32_dataloader(xte, yte, batch_size, shuffle=False)

    # Get model
    if config["all"]["model_type"] == "base_transformer":
        model_class = RecursiveCGPT  # Baseline transformer
    else:
        model_class = RecursiveNRGPT  # Energy transformer

    # Set up different model combinations to train
    n_embed = [16, 32, 64, 128, 256]
    n_head = [1, 2, 4, 8, 16]
    n_layers = [1, 2, 4, 8, 12]

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate through different model combinations and train
    project_name = config.get("all", {}).get("project_name")
    for emb, head, layer in zip(n_embed, n_head, n_layers):
        config_copy = wandb_config.copy()
        config_copy["parameters"].update({"n_embed": {"value": emb}})
        config_copy["parameters"].update({"n_head": {"value": head}})
        config_copy["parameters"].update({"n_layers": {"value": layer}})

        sweep_id = wandb.sweep(config_copy, project=project_name)
        train_wrapper = partial(
            train,
            model_class=model_class,
            device=device,
            train_data=train_dataloader,
            val_data=val_dataloader,
        )
        wandb.agent(sweep_id, function=train_wrapper, count=config["all"]["n_trials"])

        # Filter for finished runs and sort by validation loss
        print("Sweep complete. Fetching best model...")
        api = wandb.Api()

        # Evaluating
        model, best_run = get_best_sweep_model(
            api=api,
            sweep_id=sweep_id,
            project_name=project_name,
            entity=entity,
            wandb_config=config_copy,
            model_class=model_class,
        )
        test_acc = evaluate_acc(model, test_dataloader, device)
        test_loss = evaluate_cross_entropy(model, test_dataloader, device)

        # 6. Update the best run's W&B summary with final test metrics
        best_run.summary["final_test_acc"] = test_acc
        best_run.summary["final_test_loss"] = test_loss
        best_run.summary.update()

        print(f"Logged Test Acc: {test_acc:.4f} to Run {best_run.id}\n")


# ============================ Main function call ============================ #

if __name__ == "__main__":
    args_parser = ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train_listops32_model_combinations(config_path=args.config)
