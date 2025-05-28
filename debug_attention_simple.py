#!/usr/bin/env python3
"""
Simple debug script to investigate the key attention issues identified.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import torch

def debug_key_issues():
    """Debug the key attention issues."""
    print("=" * 80)
    print("DEBUGGING KEY TABPFN ATTENTION ISSUES")
    print("=" * 80)
    
    # Use synthetic data for consistent testing
    X, y = make_classification(
        n_samples=50, 
        n_features=6, 
        n_informative=5, 
        n_redundant=1, 
        n_classes=2, 
        random_state=42
    )
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test, {X.shape[1]} features")
    print(f"Feature names: {feature_names}")
    
    # Create regressor
    regressor = TabPFNRegressor(device='cpu')
    regressor.fit(X_train, y_train)
    
    print("\n" + "=" * 80)
    print("ISSUE ANALYSIS: SEQUENCE STRUCTURE AND TARGET VARIABLE")
    print("=" * 80)
    
    # Get attention for analysis
    test_sample = X_test[:1]
    pred, attention = regressor.predict(
        test_sample,
        return_attention=True,
        attention_type="features",
        attention_layer=0,
        attention_head=0
    )
    
    print(f"Input data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
    
    print(f"\nAttention tensor shape: {attention.shape}")
    seq_len, n_heads, n_features, _ = attention.shape
    
    print(f"\nSequence analysis:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Number of features in attention: {n_features}")
    print(f"  Number of input features (X): {X.shape[1]}")
    print(f"  Extra features (likely Y and special tokens): {n_features - X.shape[1]}")
    
    # Analyze what each position in the sequence represents
    print(f"\nSequence position analysis:")
    print(f"  Expected: train_samples + test_samples = {X_train.shape[0]} + {X_test.shape[0]} = {X_train.shape[0] + X_test.shape[0]}")
    print(f"  Actual sequence length: {seq_len}")
    print(f"  Difference: {seq_len - (X_train.shape[0] + X_test.shape[0])}")
    
    # The test sample should be at position X_train.shape[0]
    test_start_idx = X_train.shape[0]
    print(f"  Test sample expected position: {test_start_idx}")
    
    if test_start_idx < seq_len:
        print(f"  ✅ Test sample position is valid")
        
        # Extract attention for the test sample
        test_attention = attention[test_start_idx, 0, :, 0]  # [n_features] for head 0
        print(f"\nTest sample attention values:")
        for i, val in enumerate(test_attention):
            if i < len(feature_names):
                print(f"    Position {i} (Feature {feature_names[i]}): {val.item():.6f}")
            else:
                print(f"    Position {i} (Extra feature {i-len(feature_names)}): {val.item():.6f}")
    else:
        print(f"  ❌ Test sample position is invalid!")
    
    print("\n" + "=" * 80)
    print("ISSUE ANALYSIS: IDENTICAL VALUES IN AGGREGATION")
    print("=" * 80)
    
    # Test aggregation with different methods
    configs = [
        ("Layer 0, Head 0", {"attention_layer": 0, "attention_head": 0}),
        ("Layer 0, Head 1", {"attention_layer": 0, "attention_head": 1}),
        ("Layer 0, All Heads (Mean)", {"attention_layer": 0, "attention_aggregation": "mean"}),
        ("All Layers, All Heads (Mean)", {"attention_aggregation": "mean"}),
    ]
    
    for desc, params in configs:
        pred, attention = regressor.predict(
            test_sample,
            return_attention=True,
            attention_type="features",
            **params
        )
        
        # Extract test sample attention
        if test_start_idx < attention.shape[0]:
            test_att = attention[test_start_idx, 0, :, 0]  # First head
            unique_vals = torch.unique(test_att)
            
            print(f"\n{desc}:")
            print(f"  Unique values: {len(unique_vals)}")
            print(f"  Range: {test_att.min().item():.6f} to {test_att.max().item():.6f}")
            print(f"  First 3 values: {test_att[:3].tolist()}")
            
            # Check for the problematic case where all values are identical
            if len(unique_vals) == 1:
                print(f"  ❌ WARNING: All values are identical!")
            elif len(unique_vals) < 3:
                print(f"  ⚠️  WARNING: Very few unique values!")
            else:
                print(f"  ✅ Values are properly different")
    
    print("\n" + "=" * 80)
    print("ISSUE ANALYSIS: FEATURE INDEXING AND TARGET VARIABLE")
    print("=" * 80)
    
    # Create a comprehensive visualization
    pred, attention = regressor.predict(
        test_sample,
        return_attention=True,
        attention_type="features",
        attention_layer=0
    )
    
    if test_start_idx < attention.shape[0]:
        # Get attention matrix for test sample
        test_attention_matrix = attention[test_start_idx, :, :, 0]  # [n_heads, n_features]
        
        # Create extended feature names
        extended_feature_names = list(feature_names)
        for i in range(len(feature_names), n_features):
            extended_feature_names.append(f"Extra_{i}")
        
        print(f"Extended feature mapping:")
        for i, name in enumerate(extended_feature_names):
            print(f"  Position {i}: {name}")
        
        # Analyze attention patterns
        matrix_np = test_attention_matrix.detach().numpy()
        
        print(f"\nAttention statistics:")
        print(f"  Matrix shape: {matrix_np.shape}")
        print(f"  Mean attention per feature: {matrix_np.mean(axis=0)}")
        print(f"  Sum per head (should be ~1.0): {matrix_np.sum(axis=1)}")
        
        # Check if the last few features (likely Y-related) have different patterns
        x_features_attention = matrix_np[:, :len(feature_names)]  # Original X features
        extra_features_attention = matrix_np[:, len(feature_names):]  # Extra features (Y, etc.)
        
        print(f"\nX features attention stats:")
        print(f"  Mean: {x_features_attention.mean():.6f}")
        print(f"  Std: {x_features_attention.std():.6f}")
        
        print(f"\nExtra features attention stats:")
        print(f"  Mean: {extra_features_attention.mean():.6f}")
        print(f"  Std: {extra_features_attention.std():.6f}")
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            matrix_np,
            annot=True,
            fmt='.4f',
            xticklabels=extended_feature_names,
            yticklabels=[f'Head_{i}' for i in range(matrix_np.shape[0])],
            cmap='viridis'
        )
        plt.title(f'Feature Attention Analysis - Layer 0, All Heads')
        plt.xlabel('Features (To)')
        plt.ylabel('Attention Heads (From)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('/workspace/TabPFN/debug_attention_detailed.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved detailed visualization to: debug_attention_detailed.png")
    
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)
    
    print("Key discoveries:")
    print(f"1. Sequence includes {n_features - X.shape[1]} extra features beyond input X")
    print(f"2. These extra features likely represent target Y and special tokens")
    print(f"3. Test sample is at position {test_start_idx} in sequence")
    print(f"4. Attention sums per head: {matrix_np.sum(axis=1) if 'matrix_np' in locals() else 'N/A'}")
    print(f"5. Feature attention includes both X and Y information")

if __name__ == "__main__":
    debug_key_issues()