#!/usr/bin/env python3
"""
Debug script to understand attention extraction issues in TabPFN.
"""

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.tabpfn.regressor import TabPFNRegressor


def debug_attention_extraction():
    """Debug the attention extraction process."""
    print("="*80)
    print("DEBUG: TabPFN Attention Extraction")
    print("="*80)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use smaller subset for debugging
    X_train = X_train[:50]
    y_train = y_train[:50]
    X_test = X_test[:10]
    y_test = y_test[:10]
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Feature names: {feature_names}")
    print(f"Number of features: {len(feature_names)}")
    
    # Initialize and fit model
    print("\n2. Training TabPFN model...")
    model = TabPFNRegressor(
        n_estimators=1,
        device="cpu",
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Test different attention extraction configurations
    print("\n3. Testing attention extraction...")
    
    configs = [
        {"attention_layer": None, "attention_head": None, "attention_type": "features"},
        {"attention_layer": 0, "attention_head": None, "attention_type": "features"},
        {"attention_layer": 5, "attention_head": None, "attention_type": "features"},
        {"attention_layer": 11, "attention_head": None, "attention_type": "features"},
        {"attention_layer": None, "attention_head": None, "attention_type": "items"},
        {"attention_layer": 0, "attention_head": 0, "attention_type": "features"},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n3.{i+1}. Testing config: {config}")
        try:
            predictions, attention_matrices = model.predict(
                X_test,
                return_attention=True,
                **config
            )
            
            print(f"  Predictions shape: {predictions.shape}")
            print(f"  Number of attention matrices: {len(attention_matrices)}")
            
            for j, att_matrix in enumerate(attention_matrices):
                if att_matrix is not None:
                    print(f"  Attention matrix {j} shape: {att_matrix.shape}")
                    print(f"  Attention matrix {j} dtype: {att_matrix.dtype}")
                    print(f"  Attention matrix {j} min/max: {att_matrix.min():.6f}/{att_matrix.max():.6f}")
                    print(f"  Attention matrix {j} mean: {att_matrix.mean():.6f}")
                    
                    # Check for NaN or inf values
                    has_nan = torch.isnan(att_matrix).any()
                    has_inf = torch.isinf(att_matrix).any()
                    print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
                    
                    # Show first few values
                    if len(att_matrix.shape) >= 2:
                        print(f"  First 3x3 values:\n{att_matrix[:3, :3]}")
                else:
                    print(f"  Attention matrix {j}: None")
                    
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Test specific feature attention processing
    print("\n4. Testing feature attention processing...")
    try:
        predictions, attention_matrices = model.predict(
            X_test,
            return_attention=True,
            attention_layer=None,
            attention_head=None,
            attention_aggregation="mean",
            attention_type="features"
        )
        
        attention_matrix = attention_matrices[0]
        print(f"Raw attention matrix shape: {attention_matrix.shape}")
        
        # Process like in the visualization code
        if len(attention_matrix.shape) == 4:
            print("Processing 4D attention matrix...")
            feature_attention = attention_matrix.mean(dim=0).squeeze(-1)
            print(f"After mean(dim=0).squeeze(-1): {feature_attention.shape}")
            
            if len(feature_attention.shape) == 3:
                feature_attention = feature_attention.mean(dim=0)
                print(f"After second mean(dim=0): {feature_attention.shape}")
        else:
            feature_attention = attention_matrix.squeeze()
            print(f"After squeeze: {feature_attention.shape}")
            
        feature_attention_np = feature_attention.detach().cpu().numpy()
        print(f"Final numpy array shape: {feature_attention_np.shape}")
        print(f"Expected shape for {len(feature_names)} features + target: ({len(feature_names)+1}, {len(feature_names)+1})")
        
        # Check if dimensions match expected feature count
        expected_size = len(feature_names) + 1  # +1 for target
        if feature_attention_np.shape[0] != expected_size:
            print(f"WARNING: Shape mismatch! Got {feature_attention_np.shape[0]}, expected {expected_size}")
        
        # Show the actual values
        print(f"Attention matrix values:\n{feature_attention_np}")
        
        # Check for zero rows/columns
        zero_rows = np.all(feature_attention_np == 0, axis=1)
        zero_cols = np.all(feature_attention_np == 0, axis=0)
        print(f"Zero rows: {np.where(zero_rows)[0]}")
        print(f"Zero columns: {np.where(zero_cols)[0]}")
        
    except Exception as e:
        print(f"ERROR in feature attention processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_attention_extraction()