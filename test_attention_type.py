#!/usr/bin/env python3
"""Test script to verify attention type parameter functionality."""

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.tabpfn.regressor import TabPFNRegressor


def test_attention_types():
    """Test both feature and item attention extraction."""
    print("Loading California housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Use a smaller subset for faster testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use only first 100 samples for faster testing
    X_train = X_train[:100]
    y_train = y_train[:100]
    X_test = X_test[:20]
    y_test = y_test[:20]
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Feature names: {feature_names}")
    
    # Initialize and fit the model
    print("\nInitializing TabPFN regressor...")
    model = TabPFNRegressor(
        n_estimators=1,  # Use single estimator for faster testing
        device="cpu",
        random_state=42
    )
    
    print("Fitting model...")
    model.fit(X_train, y_train)
    
    # Test feature attention extraction
    print("\n" + "="*50)
    print("Testing FEATURE attention extraction")
    print("="*50)
    
    predictions_features, attention_features = model.predict(
        X_test,
        return_attention=True,
        attention_layer=0,  # Extract from first layer
        attention_type="features"
    )
    
    print(f"Predictions shape: {predictions_features.shape}")
    print(f"Number of attention matrices: {len(attention_features)}")
    
    if attention_features[0] is not None:
        attention_matrix = attention_features[0]
        print(f"Feature attention matrix shape: {attention_matrix.shape}")
        print(f"Feature attention matrix type: {type(attention_matrix)}")
        
        # Check if it's feature-to-feature attention
        expected_feature_shape = (len(feature_names), len(feature_names))
        if attention_matrix.shape[-2:] == expected_feature_shape:
            print("✓ Successfully extracted feature-to-feature attention!")
            print(f"  Expected shape: {expected_feature_shape}")
            print(f"  Actual shape: {attention_matrix.shape[-2:]}")
        else:
            print("✗ Feature attention shape doesn't match expected feature-to-feature shape")
            print(f"  Expected: {expected_feature_shape}")
            print(f"  Actual: {attention_matrix.shape}")
    else:
        print("✗ No feature attention matrix returned")
    
    # Test item attention extraction
    print("\n" + "="*50)
    print("Testing ITEM attention extraction")
    print("="*50)
    
    predictions_items, attention_items = model.predict(
        X_test,
        return_attention=True,
        attention_layer=0,  # Extract from first layer
        attention_type="items"
    )
    
    print(f"Predictions shape: {predictions_items.shape}")
    print(f"Number of attention matrices: {len(attention_items)}")
    
    if attention_items[0] is not None:
        attention_matrix = attention_items[0]
        print(f"Item attention matrix shape: {attention_matrix.shape}")
        print(f"Item attention matrix type: {type(attention_matrix)}")
        
        # Check if it's item-to-item attention
        total_items = len(X_train) + len(X_test)  # train + test items
        if attention_matrix.shape[-2] == total_items:
            print("✓ Successfully extracted item-to-item attention!")
            print(f"  Total items (train + test): {total_items}")
            print(f"  Attention matrix last two dims: {attention_matrix.shape[-2:]}")
        else:
            print("✗ Item attention shape doesn't match expected item-to-item shape")
            print(f"  Expected items: {total_items}")
            print(f"  Actual shape: {attention_matrix.shape}")
    else:
        print("✗ No item attention matrix returned")
    
    # Compare predictions (should be the same regardless of attention type)
    print("\n" + "="*50)
    print("Comparing predictions")
    print("="*50)
    
    if np.allclose(predictions_features, predictions_items, rtol=1e-5):
        print("✓ Predictions are identical for both attention types")
    else:
        print("✗ Predictions differ between attention types")
        print(f"  Max difference: {np.max(np.abs(predictions_features - predictions_items))}")
    
    return attention_features, attention_items


if __name__ == "__main__":
    try:
        attention_features, attention_items = test_attention_types()
        print("\n" + "="*50)
        print("Test completed successfully!")
        print("="*50)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()