# ================================== Imports ================================= #
from argparse import ArgumentParser
import yaml

import pandas as pd

from data_handling.listops32 import tokenize_listops32, separate_data_and_target


# ========================== Pipeline function call ========================== #
def tokenize_listops32_train_test(config_path) -> None:
    config = yaml.safe_load(open(config_path, "r"))
    listops32_config = config.get("listops32", {})["data_processing"]

    # Get raw data and separate
    train_df = pd.read_parquet(listops32_config["raw_train_data_path"])
    val_df = pd.read_parquet(listops32_config["raw_val_data_path"])
    test_df = pd.read_parquet(listops32_config["raw_test_data_path"])

    xtr, ytr = separate_data_and_target(train_df)
    xval, yval = separate_data_and_target(val_df)
    xtest, ytest = separate_data_and_target(test_df)

    # Tokenize data and get the dictionary
    tk_xtr, dictionary = tokenize_listops32(xtr)
    tk_xval, _ = tokenize_listops32(xval)
    tk_xtest, _ = tokenize_listops32(xtest)

    # Apply the tokenization to the target as well
    tk_ytr = pd.DataFrame(ytr.replace(dictionary))
    tk_yval = pd.DataFrame(yval.replace(dictionary))
    tk_ytest = pd.DataFrame(ytest.replace(dictionary))

    # Save data to paths
    tk_xtr.to_parquet(listops32_config["tk_train_data_path"], index=False)
    tk_ytr.to_parquet(listops32_config["tk_train_label_path"], index=False)
    tk_xval.to_parquet(listops32_config["tk_val_data_path"], index=False)
    tk_yval.to_parquet(listops32_config["tk_val_label_path"], index=False)
    tk_xtest.to_parquet(listops32_config["tk_test_data_path"], index=False)
    tk_ytest.to_parquet(listops32_config["tk_test_label_path"], index=False)

    # Save dictionary for reference
    with open(listops32_config["dictionary_path"], "w") as f:
        yaml.dump(dictionary, f)


if __name__ == "__main__":
    args_parser = ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    tokenize_listops32_train_test(config_path=args.config)
