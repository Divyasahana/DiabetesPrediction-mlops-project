import pandas as pd
from pathlib import Path


def load_data(data_path: str | Path):
    """
    Load diabetes dataset from CSV.

    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
    """
    data_path = Path(data_path)
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    return X, y