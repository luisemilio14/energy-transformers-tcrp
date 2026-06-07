# ================================== Imports ================================= #
# Standard library imports
from argparse import ArgumentParser
from functools import partial
import os
import yaml

# Third-party imports
import optuna
from optuna.pruners import MedianPruner
import pandas as pd
import torch

# Custom imports
from data_handling.listops32 import generate_listops32_dataloader
from training.training import hyperparameter_optimization
from energy_transformers.baseline_transformer import RecursiveCGPT
from energy_transformers.energy_transformer import RecursiveNRGPT


# ========================== Pipeline function call ========================== #
def train_listops32_model(config_path) -> None:
    config = yaml.safe_load(open(config_path, "r"))
    optuna_config = config.get("optuna", {})
    data_config = config.get("listops32", {})["data_processing"]
    train_config = config.get("listops32", {})["train_model"]
    if config.get("all", {}).get("model_type") == "base_transformer":
        model_config = config.get("models", {})["base_transformer"]
        model_class = RecursiveCGPT
    elif config.get("all", {}).get("model_type") == "energy_transformer":
        model_class = RecursiveNRGPT
        model_config = config.get("models", {})["energy_transformer"]
    else:
        raise ValueError("Invalid model type specified in config")

    # Load tokenized data
    xtr = pd.read_parquet(data_config["tk_train_data_path"])
    ytr = pd.read_parquet(data_config["tk_train_label_path"])
    xval = pd.read_parquet(data_config["tk_val_data_path"])
    yval = pd.read_parquet(data_config["tk_val_label_path"])

    batch_size = train_config["batch_size"]
    train_dataloader = generate_listops32_dataloader(xtr, ytr, batch_size, shuffle=True)
    val_dataloader = generate_listops32_dataloader(
        xval, yval, batch_size, shuffle=False
    )

    # --- Parsing additional config parameters --- #
    # Add data-specific parameters to training config
    train_config["total_dataset_samples"] = len(xtr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config["device"] = device

    # Add data-specific parameters to model config
    model_config["sequence_len"] = xtr.shape[1]
    model_config["vocab_size"] = len(
        pd.concat([xtr, xval], ignore_index=True).stack().unique()
    )
    model_config["n_classes"] = len(
        pd.concat([ytr, yval], ignore_index=True).stack().unique()
    )

    # --- Optuna optimization setup --- #
    # Create pruner
    # TODO: create factory, add more methods
    pruner = MedianPruner(n_warmup_steps=optuna_config["pruner"]["n_warmup_steps"])

    # Create study
    study = optuna.create_study(
        study_name=optuna_config["study_name"],
        direction="minimize",  # Loss (CrossEntropy)
        load_if_exists=True,
        pruner=pruner,
        # storage=os.getenv("STORAGE_NAME"),
    )
    # Define objective function for Optuna optimization
    objective_fn = partial(
        hyperparameter_optimization,
        model_class=model_class,
        train_data=train_dataloader,
        val_data=val_dataloader,
        train_params=train_config,
        model_params=model_config,
    )
    # Start optimizing gogogo
    study.optimize(
        objective_fn,
        n_trials=optuna_config["n_trials"],
    )


# ============================ Main function call ============================ #

if __name__ == "__main__":
    args_parser = ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train_listops32_model(config_path=args.config)
