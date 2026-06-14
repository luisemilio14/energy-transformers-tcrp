# ================================== Imports ================================= #
# Standard library imports
from argparse import ArgumentParser
from functools import partial

# Third-party imports
import pandas as pd
import wandb
import torch
import yaml

# Custom imports
from data_handling.listops32 import generate_listops32_dataloader
from training.training_wandb import train
from energy_transformers.baseline_transformer import RecursiveCGPT
from energy_transformers.energy_transformer import RecursiveNRGPT


# ========================== Pipeline function call ========================== #
def train_listops32_model_wandb(config_path) -> None:
    config = yaml.safe_load(open(config_path, "r"))
    data_config = config.get("listops32", {})["data_processing"]
    wandb_path = config.get("all", {}).get("wandb_config_file", "wandb_config.yaml")
    wandb_config = yaml.safe_load(open(wandb_path, "r"))

    # Create dataloaders
    xtr = pd.read_parquet(data_config["tk_train_data_path"])
    ytr = pd.read_parquet(data_config["tk_train_label_path"])
    xval = pd.read_parquet(data_config["tk_val_data_path"])
    yval = pd.read_parquet(data_config["tk_val_label_path"])
    # TODO: eventually pass batch size as an opt param
    batch_size = wandb_config["parameters"]["batch_size"]["value"]
    train_dataloader = generate_listops32_dataloader(xtr, ytr, batch_size, shuffle=True)
    val_dataloader = generate_listops32_dataloader(
        xval, yval, batch_size, shuffle=False
    )

    # Get model
    if config["all"]["model_type"] == "base_transformer":
        model_class = RecursiveCGPT  # Baseline transformer
    else:
        model_class = RecursiveNRGPT  # Energy transformer

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start execution - set up a sweep
    project_name = config.get("all", {}).get("project_name")
    sweep_id = wandb.sweep(wandb_config, project=project_name)
    train_wrapper = partial(
        train,
        model_class=model_class,
        device=device,
        train_data=train_dataloader,
        val_data=val_dataloader,
    )
    wandb.agent(sweep_id, function=train_wrapper, count=config["all"]["n_trials"])


# ============================ Main function call ============================ #

if __name__ == "__main__":
    args_parser = ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train_listops32_model_wandb(config_path=args.config)
