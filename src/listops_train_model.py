# ================================== Imports ================================= #
# Standard library imports
from argparse import ArgumentParser
import os
import yaml

# Third-party imports
import optuna
from optuna.pruners import MedianPruner
import pandas as pd


# ========================== Pipeline function call ========================== #
def train_listops32_model(config_path) -> None:
    config = yaml.safe_load(open(config_path, "r"))
    optuna_config = config.get("optuna", {})
    data_config = config.get("listops32", {})["data_processing"]
    train_config = config.get("listops32", {})["train_model"]
    model_config = config.get("listops32", {})["model_config"]

    # Load tokenized data
    xtr = pd.read_parquet(data_config["tk_train_data_path"])
    ytr = pd.read_parquet(data_config["tk_train_label_path"])
    xval = pd.read_parquet(data_config["tk_val_data_path"])
    yval = pd.read_parquet(data_config["tk_val_label_path"])

    # Optuna: create pruner
    # TODO: create factory, add more methods
    pruner = MedianPruner(n_warmup_steps=optuna_config["pruner_warmup_steps"])

    # Create study
    study = optuna.create_study(
        study_name=optuna_config["study_name"],
        direction="minimize",
        load_if_exists=True,
        pruner=pruner,
        storage=os.getenv("STORAGE_NAME"),
    )

    # Start optimizing
    # TODO


# ============================ Main function call ============================ #

if __name__ == "__main__":
    args_parser = ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train_listops32_model(config_path=args.config)
