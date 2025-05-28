#!/usr/bin/env python3

"""
Debug script to understand attention extraction issues.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor

def debug_attention_extraction():
    """Debug attention extraction to understand the issues."""
    
    print("Creating simple regression dataset...")
    X, y = make_regression(n_samples=50, n_features=4, noise=0.1, random_state=42)
    
    # Use a smaller subset for debugging
    X_train = X[:30]
    y_train = y[:30]
    X_test = X[30:35]  # Just 5 test samples
    y_test = y[30:35]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    # Initialize TabPFN regressor
    regressor = TabPFNRegressor(device='cpu')
    
    # Fit the model
    print("Fitting TabPFN regressor...")
    regressor.fit(X_train_scaled, y_train)
    
    # Test 1: Normal prediction
    print("\nTest 1: Normal prediction...")
    predictions_normal = regressor.predict(X_test_scaled)
    print(f"Normal predictions shape: {predictions_normal.shape}")
    print(f"Normal predictions: {predictions_normal}")
    
    # Test 2: Extract attention without any filtering
    print("\nTest 2: Extract attention without filtering...")
    predictions_attn, attention_raw = regressor.predict(
        X_test_scaled, 
        return_attention=True
    )
    
    print(f"Predictions with attention shape: {predictions_attn.shape}")
    print(f"Predictions match: {np.allclose(predictions_normal, predictions_attn, rtol=1e-5)}")
    
    if attention_raw is not None:
        if isinstance(attention_raw, list):
            print(f"Attention is a list with {len(attention_raw)} elements")
            for i, attn in enumerate(attention_raw):
                if attn is not None:
                    print(f"  Element {i}: shape {attn.shape}, min/max: {attn.min():.4f}/{attn.max():.4f}")
                    print(f"  Element {i}: dtype {attn.dtype}, device {attn.device}")
                else:
                    print(f"  Element {i}: None")
        else:
            print(f"Attention shape: {attention_raw.shape}")
            print(f"Attention min/max: {attention_raw.min():.4f}/{attention_raw.max():.4f}")
    else:
        print("No attention extracted")
    
    # Test 3: Extract attention with specific layer
    print("\nTest 3: Extract attention from layer 0...")
    predictions_layer0, attention_layer0 = regressor.predict(
        X_test_scaled, 
        return_attention=True,
        attention_layer=0
    )
    
    print(f"Layer 0 predictions match: {np.allclose(predictions_normal, predictions_layer0, rtol=1e-5)}")
    
    if attention_layer0 is not None:
        if isinstance(attention_layer0, list):
            print(f"Layer 0 attention is a list with {len(attention_layer0)} elements")
            for i, attn in enumerate(attention_layer0):
                if attn is not None:
                    print(f"  Element {i}: shape {attn.shape}, min/max: {attn.min():.4f}/{attn.max():.4f}")
                else:
                    print(f"  Element {i}: None")
        else:
            print(f"Layer 0 attention shape: {attention_layer0.shape}")
            print(f"Layer 0 attention min/max: {attention_layer0.min():.4f}/{attention_layer0.max():.4f}")
    
    # Test 4: Extract attention with specific head
    print("\nTest 4: Extract attention from head 0...")
    predictions_head0, attention_head0 = regressor.predict(
        X_test_scaled, 
        return_attention=True,
        attention_head=0
    )
    
    print(f"Head 0 predictions match: {np.allclose(predictions_normal, predictions_head0, rtol=1e-5)}")
    
    if attention_head0 is not None:
        if isinstance(attention_head0, list):
            print(f"Head 0 attention is a list with {len(attention_head0)} elements")
            for i, attn in enumerate(attention_head0):
                if attn is not None:
                    print(f"  Element {i}: shape {attn.shape}, min/max: {attn.min():.4f}/{attn.max():.4f}")
                else:
                    print(f"  Element {i}: None")
        else:
            print(f"Head 0 attention shape: {attention_head0.shape}")
            print(f"Head 0 attention min/max: {attention_head0.min():.4f}/{attention_head0.max():.4f}")
    
    # Test 5: Extract attention with different aggregation
    print("\nTest 5: Extract attention with max aggregation...")
    predictions_max, attention_max = regressor.predict(
        X_test_scaled, 
        return_attention=True,
        attention_aggregation="max"
    )
    
    print(f"Max aggregation predictions match: {np.allclose(predictions_normal, predictions_max, rtol=1e-5)}")
    
    if attention_max is not None:
        if isinstance(attention_max, list):
            print(f"Max aggregation attention is a list with {len(attention_max)} elements")
            for i, attn in enumerate(attention_max):
                if attn is not None:
                    print(f"  Element {i}: shape {attn.shape}, min/max: {attn.min():.4f}/{attn.max():.4f}")
                else:
                    print(f"  Element {i}: None")
        else:
            print(f"Max aggregation attention shape: {attention_max.shape}")
            print(f"Max aggregation attention min/max: {attention_max.min():.4f}/{attention_max.max():.4f}")
    
    # Test 6: Extract attention with attention_type parameter
    print("\nTest 6: Extract feature attention...")
    try:
        predictions_features, attention_features = regressor.predict(
            X_test_scaled, 
            return_attention=True,
            attention_type="features"
        )
        
        print(f"Feature attention predictions match: {np.allclose(predictions_normal, predictions_features, rtol=1e-5)}")
        
        if attention_features is not None:
            if isinstance(attention_features, list):
                print(f"Feature attention is a list with {len(attention_features)} elements")
                for i, attn in enumerate(attention_features):
                    if attn is not None:
                        print(f"  Element {i}: shape {attn.shape}, min/max: {attn.min():.4f}/{attn.max():.4f}")
                    else:
                        print(f"  Element {i}: None")
            else:
                print(f"Feature attention shape: {attention_features.shape}")
                print(f"Feature attention min/max: {attention_features.min():.4f}/{attention_features.max():.4f}")
    except Exception as e:
        print(f"Error extracting feature attention: {e}")
    
    print("\nTest 7: Extract item attention...")
    try:
        predictions_items, attention_items = regressor.predict(
            X_test_scaled, 
            return_attention=True,
            attention_type="items"
        )
        
        print(f"Item attention predictions match: {np.allclose(predictions_normal, predictions_items, rtol=1e-5)}")
        
        if attention_items is not None:
            if isinstance(attention_items, list):
                print(f"Item attention is a list with {len(attention_items)} elements")
                for i, attn in enumerate(attention_items):
                    if attn is not None:
                        print(f"  Element {i}: shape {attn.shape}, min/max: {attn.min():.4f}/{attn.max():.4f}")
                    else:
                        print(f"  Element {i}: None")
            else:
                print(f"Item attention shape: {attention_items.shape}")
                print(f"Item attention min/max: {attention_items.min():.4f}/{attention_items.max():.4f}")
    except Exception as e:
        print(f"Error extracting item attention: {e}")

if __name__ == "__main__":
    debug_attention_extraction()