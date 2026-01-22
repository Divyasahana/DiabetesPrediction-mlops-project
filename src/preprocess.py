"""
Minimal preprocessing: train/test split and optional scaling.
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(X, y, test_size=0.2, random_state=42, scale=True):
    """
    Split data into train/test and optionally scale features.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Fraction for test set
        random_state (int): Seed for reproducibility
        scale (bool): Whether to standardize features
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"✓ Train/test split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    # Optional scaling
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("✓ Applied StandardScaler")
    
    return X_train, X_test, y_train, y_test