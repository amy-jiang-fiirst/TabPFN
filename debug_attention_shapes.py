#!/usr/bin/env python3
"""
Debug attention shapes from different ensemble configurations.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor

def debug_attention_shapes():
    """Debug attention shapes from different ensemble configurations."""
    print("Debugging attention shapes...")
    
    # Generate small test data
    X, y = make_classification(n_samples=20, n_features=3, n_informative=2, n_redundant=1, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Create regressor
    regressor = TabPFNRegressor(device='cpu')
    regressor.fit(X_train, y_train)
    
    # Test different attention types
    for attention_type in ["features", "items"]:
        print(f"\n=== Testing {attention_type} attention ===")
        pred, attention = regressor.predict(X_test, return_attention=True, attention_layer=0, attention_type=attention_type)
        
        print(f"Prediction shape: {pred.shape}")
        print(f"Attention type: {type(attention)}")
        if hasattr(attention, 'shape'):
            print(f"Attention shape: {attention.shape}")
        elif isinstance(attention, list):
            print(f"Attention is list with {len(attention)} elements")
            for i, att in enumerate(attention):
                if hasattr(att, 'shape'):
                    print(f"  Element {i}: shape={att.shape}")

if __name__ == "__main__":
    debug_attention_shapes()