"""
Load CSV data and return features and target.
"""
import pandas as pd
from pathlib import Path


def load_data(filepath='data/diabetes.csv'):
    """
    Load diabetes dataset from CSV.
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
    """
    # Ensure path exists
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Read CSV
    df = pd.read_csv(filepath)
    
    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    print(f"✓ Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y