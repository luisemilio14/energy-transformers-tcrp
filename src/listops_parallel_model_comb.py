# ================================== Imports ================================= #
# Standard library imports
from argparse import ArgumentParser
from functools import partial
import itertools
import os
import multiprocessing as mp

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
import wandb
import torch
import yaml

# Custom imports
from data_handling.listops32 import generate_listops32_dataloader
from evaluation.evaluate import evaluate_cross_entropy, evaluate_acc
from energy_transformers.baseline_transformer import RecursiveCGPT
from energy_transformers.energy_transformer import RecursiveNRGPT
from utils.model_loader import get_best_sweep_model


# ============================ Auxiliary functions =========================== #
def run_agent_on_gpu(sweep_id, project_name, gpu_id, trials, config_path, wandb_config):
    """Isolated process that runs a W&B agent on a specific GPU."""
    # 1. Hide the other GPU from this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Re-import libraries
    import torch
    from data_handling.listops32 import generate_listops32_dataloader
    from training.training_wandb import train
    import pandas as pd
    import yaml

    # Test - limiting cpu usage for each agent
    torch.set_num_threads(2)

    # Get gpu and config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(open(config_path, "r"))
    data_config = config.get("listops32", {})["data_processing"]

    # Load parquets
    # Create dataloaders
    xtr = pd.read_parquet(data_config["tk_train_data_path"])
    ytr = pd.read_parquet(data_config["tk_train_label_path"])
    xval = pd.read_parquet(data_config["tk_val_data_path"])
    yval = pd.read_parquet(data_config["tk_val_label_path"])

    batch_size = wandb_config["parameters"]["batch_size"]["value"]
    if batch_size == 0:
        # If zero, use full training set for batch size
        batch_size = ytr.shape[0]
    train_dataloader = generate_listops32_dataloader(
        xtr,
        ytr,
        batch_size,
        shuffle=True,
        device=device,
        num_workers=1,
    )
    val_dataloader = generate_listops32_dataloader(
        xval,
        yval,
        yval.shape[0],  # Use full val set for evaluation
        shuffle=False,
        device=device,
        num_workers=1,
    )

    # Get model
    if config["all"]["model_type"] == "base_transformer":
        model_class = RecursiveCGPT  # Baseline transformer
    else:
        model_class = RecursiveNRGPT  # Energy transformer

    # Create wrapper and run trial
    train_wrapper = partial(
        train,
        model_class=model_class,
        device=device,
        train_data=train_dataloader,
        val_data=val_dataloader,
        print_batch_interval=max(1, len(train_dataloader) // 4),
    )
    wandb.agent(sweep_id, function=train_wrapper, count=trials, project=project_name)


# ========================== Pipeline function call ========================== #
def train_listops32_parallel_model_comb(config_path) -> None:
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
    if config["all"]["model_type"] == "base_transformer":
        model_class = RecursiveCGPT  # Baseline transformer
    else:
        model_class = RecursiveNRGPT  # Energy transformer

    # Get device
    if config["all"]["device"] == "cuda":
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    else:
        device = torch.device("cpu")

    # Load test data - train and val loaded separately per-GPU
    xte = pd.read_parquet(data_config["tk_test_data_path"])
    yte = pd.read_parquet(data_config["tk_test_label_path"])

    # Set up different model combinations to train
    # n_embed = [16, 32, 64, 128, 256, 512]
    n_embed = [64, 128, 256, 512]
    # n_head = [1, 2, 4, 8, 16, 32]
    # n_head = [1, 2, 4, 8, 16, 32]
    n_head = [1, 2, 4]
    # n_layers = [1, 2, 4, 8]
    n_layers = [1]

    # Iterate through different model combinations and train
    project_name = config.get("all", {}).get("project_name")
    for emb, head, layer in itertools.product(n_embed, n_head, n_layers):
        config_copy = wandb_config.copy()
        config_copy["parameters"].update({"n_embed": {"value": emb}})
        config_copy["parameters"].update({"n_head": {"value": head}})
        config_copy["parameters"].update({"n_layers": {"value": layer}})

        # Configure sweep and trials per GPU
        sweep_id = wandb.sweep(config_copy, project=project_name)
        total_trials = config["all"]["n_trials"]
        n_gpus = config["all"].get("n_gpus")
        trials_per_gpu = total_trials // n_gpus

        # Dispatch agents to GPUs
        ctx = mp.get_context("spawn")
        processes = []
        for i in range(n_gpus):
            # Give any leftover trials to the last GPU
            remainder = (total_trials % n_gpus) if (i == n_gpus - 1) else 0
            agent_trials = trials_per_gpu + remainder

            # Skip spawning an agent if it has 0 trials assigned
            if agent_trials <= 0:
                continue

            p = ctx.Process(
                target=run_agent_on_gpu,
                args=(
                    sweep_id,
                    project_name,
                    i,
                    agent_trials,
                    config_path,
                    config_copy,
                ),
            )
            p.start()
            processes.append(p)

        # Wait for all dynamically spawned agents to finish
        for p in processes:
            p.join()

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
            device=device,
        )
        test_dataloader = generate_listops32_dataloader(
            xte,
            yte,
            yte.shape[0],  # Use full test set for evaluation
            shuffle=False,
            device=device,
            num_workers=1,
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

    train_listops32_parallel_model_comb(config_path=args.config)
