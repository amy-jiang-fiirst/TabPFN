#!/usr/bin/env python3
"""
Debug the number of ensemble configurations.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor

def debug_ensemble_configs():
    """Debug the number of ensemble configurations."""
    print("Debugging ensemble configurations...")
    
    # Generate small test data
    X, y = make_classification(n_samples=20, n_features=3, n_informative=2, n_redundant=1, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Create regressor
    regressor = TabPFNRegressor(device='cpu')
    regressor.fit(X_train, y_train)
    
    # Check ensemble configurations
    print(f"Number of ensemble configurations: {len(regressor.executor_.ensemble_configs)}")
    
    # Test attention extraction
    pred, attention = regressor.predict(X_test, return_attention=True, attention_layer=0)
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Attention type: {type(attention)}")
    print(f"Number of attention matrices: {len(attention) if isinstance(attention, list) else 'N/A'}")
    
    if isinstance(attention, list) and len(attention) > 0:
        first_att = attention[0]
        print(f"First attention type: {type(first_att)}")
        if hasattr(first_att, 'shape'):
            print(f"First attention shape: {first_att.shape}")

if __name__ == "__main__":
    debug_ensemble_configs()