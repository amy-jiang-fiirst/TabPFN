#!/usr/bin/env python3
"""
Debug head dimension in attention extraction.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor

def debug_head_dimension():
    """Debug head dimension in attention extraction."""
    print("Debugging head dimension...")
    
    # Generate small test data
    X, y = make_classification(n_samples=20, n_features=3, n_informative=2, n_redundant=1, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Create regressor
    regressor = TabPFNRegressor(device='cpu')
    regressor.fit(X_train, y_train)
    
    # Test different head configurations for features attention
    print("\n=== Features Attention ===")
    
    # All heads
    pred, attention_all = regressor.predict(
        X_test, 
        return_attention=True, 
        attention_layer=0,
        attention_head=None,
        attention_type="features"
    )
    print(f"All heads - Attention shape: {attention_all.shape}")
    
    # Specific head 0
    pred, attention_h0 = regressor.predict(
        X_test, 
        return_attention=True, 
        attention_layer=0,
        attention_head=0,
        attention_type="features"
    )
    print(f"Head 0 - Attention shape: {attention_h0.shape}")
    
    # Specific head 1
    pred, attention_h1 = regressor.predict(
        X_test, 
        return_attention=True, 
        attention_layer=0,
        attention_head=1,
        attention_type="features"
    )
    print(f"Head 1 - Attention shape: {attention_h1.shape}")
    
    # Check if attention values are different
    if attention_all.shape == attention_h0.shape == attention_h1.shape:
        print("\nChecking if attention values are different:")
        print(f"All heads vs Head 0 - Same values: {torch.allclose(attention_all, attention_h0)}")
        print(f"Head 0 vs Head 1 - Same values: {torch.allclose(attention_h0, attention_h1)}")
        
        # Check some sample values
        print(f"\nSample values:")
        print(f"All heads [0,0,0,0]: {attention_all[0,0,0,0].item():.6f}")
        print(f"Head 0 [0,0,0,0]: {attention_h0[0,0,0,0].item():.6f}")
        print(f"Head 1 [0,0,0,0]: {attention_h1[0,0,0,0].item():.6f}")

if __name__ == "__main__":
    import torch
    debug_head_dimension()