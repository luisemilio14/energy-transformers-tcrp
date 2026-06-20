"""
Script for downloading and handling the ListOps dataset.
"""

# ================================== Imports ================================= #
import pandas as pd
import torch


# ============= Main tokenization function for ListOps32 dataset ============= #
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


# ========================== Data loading utilities ========================== #
class ListOps32Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.long)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_listops32_dataloader(X, y, batch_size, shuffle, num_workers=8):
    dataset = ListOps32Dataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


# ============================ Auxiliary functions =========================== #
def separate_data_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Separate target and data
    y = df["Target"]
    X = df.drop(columns=["Target"])
    return X, y
