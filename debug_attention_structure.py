#!/usr/bin/env python3
"""
Debug the attention structure to understand the nested list issue.
"""

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor

def debug_attention_structure():
    """Debug the attention structure."""
    print("Debugging attention structure...")
    
    # Generate small test data
    X, y = make_classification(n_samples=20, n_features=3, n_informative=2, n_redundant=1, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Create regressor
    regressor = TabPFNRegressor(device='cpu')
    regressor.fit(X_train, y_train)
    
    # Test layer-specific extraction
    print("\n=== Testing Layer 0 ===")
    pred_layer_0, attention_layer_0 = regressor.predict(X_test, return_attention=True, attention_layer=0)
    
    print(f"Prediction shape: {pred_layer_0.shape}")
    print(f"Attention type: {type(attention_layer_0)}")
    
    if isinstance(attention_layer_0, list):
        print(f"Attention is list with {len(attention_layer_0)} elements")
        for i, att in enumerate(attention_layer_0):
            print(f"  Element {i}: type={type(att)}")
            if isinstance(att, list):
                print(f"    Nested list with {len(att)} elements")
                for j, sub_att in enumerate(att):
                    if hasattr(sub_att, 'shape'):
                        print(f"      Sub-element {j}: shape={sub_att.shape}")
                    else:
                        print(f"      Sub-element {j}: type={type(sub_att)}")
            elif hasattr(att, 'shape'):
                print(f"    Shape: {att.shape}")
    elif hasattr(attention_layer_0, 'shape'):
        print(f"Attention shape: {attention_layer_0.shape}")
    
    # Test all layers
    print("\n=== Testing All Layers ===")
    pred_all, attention_all = regressor.predict(X_test, return_attention=True)
    
    print(f"Prediction shape: {pred_all.shape}")
    print(f"Attention type: {type(attention_all)}")
    
    if isinstance(attention_all, list):
        print(f"Attention is list with {len(attention_all)} elements")
        for i, att in enumerate(attention_all):
            print(f"  Element {i}: type={type(att)}")
            if isinstance(att, list):
                print(f"    Nested list with {len(att)} elements")
            elif hasattr(att, 'shape'):
                print(f"    Shape: {att.shape}")

if __name__ == "__main__":
    debug_attention_structure()