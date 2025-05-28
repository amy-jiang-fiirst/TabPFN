#!/usr/bin/env python3
"""
Test head-specific attention extraction.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor

def test_head_specific():
    """Test head-specific attention extraction."""
    print("Testing head-specific attention extraction...")
    
    # Generate small test data
    X, y = make_classification(n_samples=20, n_features=3, n_informative=2, n_redundant=1, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Create regressor
    regressor = TabPFNRegressor(device='cpu')
    regressor.fit(X_train, y_train)
    
    # Test different head configurations
    test_cases = [
        {"attention_layer": 0, "attention_head": 0, "desc": "Layer 0, Head 0"},
        {"attention_layer": 0, "attention_head": 1, "desc": "Layer 0, Head 1"},
        {"attention_layer": 0, "attention_head": None, "desc": "Layer 0, All Heads"},
        {"attention_layer": None, "attention_head": 0, "desc": "All Layers, Head 0"},
        {"attention_layer": None, "attention_head": None, "desc": "All Layers, All Heads"},
    ]
    
    for case in test_cases:
        print(f"\n=== {case['desc']} ===")
        try:
            pred, attention = regressor.predict(
                X_test, 
                return_attention=True, 
                attention_layer=case['attention_layer'],
                attention_head=case['attention_head'],
                attention_type="features"
            )
            
            print(f"Prediction shape: {pred.shape}")
            print(f"Attention type: {type(attention)}")
            if hasattr(attention, 'shape'):
                print(f"Attention shape: {attention.shape}")
            elif isinstance(attention, list):
                print(f"Attention is list with {len(attention)} elements")
            else:
                print(f"Attention: {attention}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_head_specific()