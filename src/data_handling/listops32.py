"""
Script for downloading and handling the ListOps dataset.
"""

# ================================== Imports ================================= #
import pandas as pd


def tokenize_listops32(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Extract characters and make a dataframe out of them
    regex_pat = r"[A-Z]+|\d+|[\[\](),]"
    X = pd.DataFrame(df["Source"].str.findall(regex_pat).to_list())
    X.columns = [f"tk_{i}" for i in range(X.shape[1])]
    X.fillna("nan", inplace=True)

    # Apply tokenization/converison to integers
    vocab = sorted(set(X.values.ravel()))
    dictionary = {token: idx for idx, token in enumerate(vocab)}
    tkX = X.replace(dictionary)

    return tkX, dictionary


def separate_data_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Separate target and data
    y = df["Target"]
    X = df.drop(columns=["Target"])
    return X, y
