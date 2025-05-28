#!/usr/bin/env python3
"""
Debug script to investigate attention issues:
1. Identical values in aggregated attention
2. Target variable Y involvement in feature attention
3. Empty attention for target/other variables
4. Feature name indexing alignment
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import torch

def debug_attention_issues():
    """Debug all attention-related issues."""
    print("=" * 80)
    print("DEBUGGING TABPFN ATTENTION ISSUES")
    print("=" * 80)
    
    # Use a real dataset with known feature names for better debugging
    try:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        X, y = data.data[:100], data.target[:100]  # Limit to 100 samples
        feature_names = list(data.feature_names)
        print(f"Using California Housing dataset (limited to 100 samples)")
    except:
        # Fallback to synthetic data
        X, y = make_classification(
            n_samples=50, 
            n_features=6, 
            n_informative=5, 
            n_redundant=1, 
            n_classes=2, 
            random_state=42
        )
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        print(f"Using synthetic dataset")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test, {X.shape[1]} features")
    print(f"Feature names: {feature_names}")
    print(f"Train X shape: {X_train.shape}, y shape: {y_train.shape}")
    print(f"Test X shape: {X_test.shape}, y shape: {y_test.shape}")
    
    # Create regressor
    regressor = TabPFNRegressor(device='cpu')
    regressor.fit(X_train, y_train)
    
    print("\n" + "=" * 80)
    print("ISSUE 1: INVESTIGATING IDENTICAL VALUES IN AGGREGATED ATTENTION")
    print("=" * 80)
    
    # Test different aggregation methods
    test_sample = X_test[:1]  # Single test sample
    
    configs = [
        ("Layer 0, Head 0", {"attention_layer": 0, "attention_head": 0}),
        ("Layer 0, Head 1", {"attention_layer": 0, "attention_head": 1}),
        ("Layer 0, All Heads (Mean)", {"attention_layer": 0, "attention_aggregation": "mean"}),
        ("Layer 0, All Heads (Max)", {"attention_layer": 0, "attention_aggregation": "max"}),
        ("All Layers, All Heads (Mean)", {"attention_aggregation": "mean"}),
    ]
    
    attention_results = {}
    
    for desc, params in configs:
        pred, attention = regressor.predict(
            test_sample,
            return_attention=True,
            attention_type="features",
            **params
        )
        
        print(f"\n{desc}:")
        print(f"  Attention shape: {attention.shape}")
        print(f"  Attention dtype: {attention.dtype}")
        
        # Extract attention matrix for analysis
        if attention.dim() == 4:  # [seq_len, n_heads, n_features, 1]
            # Get attention for the test sample (last position in sequence)
            test_start_idx = X_train.shape[0]
            if test_start_idx < attention.shape[0]:
                att_matrix = attention[test_start_idx, :, :, 0]  # [n_heads, n_features]
                print(f"  Extracted matrix shape: {att_matrix.shape}")
                
                # Check for identical values
                unique_values = torch.unique(att_matrix)
                print(f"  Unique values count: {len(unique_values)}")
                print(f"  Value range: {att_matrix.min().item():.6f} to {att_matrix.max().item():.6f}")
                print(f"  Sample values: {att_matrix[0, :3].tolist()}")
                
                # Check if all values are identical
                if len(unique_values) == 1:
                    print(f"  ❌ WARNING: All values are identical ({unique_values[0].item():.6f})")
                else:
                    print(f"  ✅ Values are different")
                
                attention_results[desc] = att_matrix.detach().numpy()
            else:
                print(f"  ❌ ERROR: test_start_idx {test_start_idx} >= attention.shape[0] {attention.shape[0]}")
        else:
            print(f"  ❌ ERROR: Unexpected attention shape: {attention.shape}")
    
    print("\n" + "=" * 80)
    print("ISSUE 2: TARGET VARIABLE Y INVOLVEMENT IN FEATURE ATTENTION")
    print("=" * 80)
    
    # Investigate the sequence structure in TabPFN
    pred, attention = regressor.predict(
        test_sample,
        return_attention=True,
        attention_type="features",
        attention_layer=0,
        attention_head=0
    )
    
    print(f"Full attention shape: {attention.shape}")
    print(f"Expected sequence length: train_samples + test_samples = {X_train.shape[0]} + {X_test.shape[0]} = {X_train.shape[0] + X_test.shape[0]}")
    print(f"Actual sequence length: {attention.shape[0]}")
    
    # Analyze the sequence structure
    seq_len = attention.shape[0]
    n_heads = attention.shape[1] 
    n_features = attention.shape[2]
    
    print(f"\nSequence analysis:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Number of features: {n_features}")
    print(f"  Expected features (X): {X.shape[1]}")
    print(f"  Does sequence include Y? {n_features > X.shape[1]}")
    
    if n_features > X.shape[1]:
        print(f"  Extra features (likely Y): {n_features - X.shape[1]}")
        print(f"  Feature breakdown: {X.shape[1]} X features + {n_features - X.shape[1]} Y features = {n_features} total")
    
    print("\n" + "=" * 80)
    print("ISSUE 3: EMPTY ATTENTION FOR TARGET AND OTHER VARIABLES")
    print("=" * 80)
    
    # Analyze attention patterns across the sequence
    test_start_idx = X_train.shape[0]
    
    if test_start_idx < seq_len:
        # Get attention for test sample
        test_attention = attention[test_start_idx, 0, :, 0]  # [n_features] for head 0
        
        print(f"Test sample attention analysis:")
        print(f"  Test sample index in sequence: {test_start_idx}")
        print(f"  Attention values: {test_attention.tolist()}")
        print(f"  Non-zero count: {(test_attention != 0).sum().item()}/{len(test_attention)}")
        
        # Check attention for each feature position
        for i, val in enumerate(test_attention):
            feature_name = feature_names[i] if i < len(feature_names) else f"Extra_{i}"
            print(f"    Feature {i} ({feature_name}): {val.item():.6f}")
        
        # Check if there are empty rows/columns
        full_matrix = attention[test_start_idx, 0, :, :]  # [n_features, 1]
        empty_features = (full_matrix.sum(dim=1) == 0).sum().item()
        print(f"  Empty feature rows: {empty_features}/{n_features}")
        
    print("\n" + "=" * 80)
    print("ISSUE 4: FEATURE NAME INDEXING ALIGNMENT")
    print("=" * 80)
    
    # Create a detailed visualization to check indexing
    if test_start_idx < seq_len:
        test_attention_matrix = attention[test_start_idx, :, :, 0]  # [n_heads, n_features]
        
        print(f"Creating detailed feature attention visualization...")
        print(f"Matrix shape: {test_attention_matrix.shape}")
        
        # Create extended feature names including potential Y
        extended_feature_names = list(feature_names)
        if n_features > len(feature_names):
            for i in range(len(feature_names), n_features):
                extended_feature_names.append(f"Target_Y_{i-len(feature_names)}")
        
        print(f"Extended feature names: {extended_feature_names}")
        
        # Create heatmap with proper labels
        plt.figure(figsize=(12, 8))
        
        # Convert to numpy for visualization
        matrix_np = test_attention_matrix.detach().numpy()
        
        # Create head labels
        head_labels = [f'Head_{i}' for i in range(matrix_np.shape[0])]
        
        sns.heatmap(
            matrix_np,
            annot=True,
            fmt='.6f',
            xticklabels=extended_feature_names,
            yticklabels=head_labels,
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.title(f'Feature Attention Analysis - Test Sample {test_start_idx}')
        plt.xlabel('Features (To)')
        plt.ylabel('Attention Heads (From)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('/workspace/TabPFN/debug_feature_attention_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved detailed analysis to: debug_feature_attention_analysis.png")
        
        # Statistical analysis
        print(f"\nStatistical analysis:")
        print(f"  Mean attention per feature: {matrix_np.mean(axis=0)}")
        print(f"  Std attention per feature: {matrix_np.std(axis=0)}")
        print(f"  Mean attention per head: {matrix_np.mean(axis=1)}")
        print(f"  Std attention per head: {matrix_np.std(axis=1)}")
        
    print("\n" + "=" * 80)
    print("ISSUE 5: INVESTIGATING AGGREGATION FUNCTION")
    print("=" * 80)
    
    # Debug the aggregation function directly
    print("Testing aggregation function behavior...")
    
    # Monkey patch to capture raw attention
    original_aggregate = regressor._aggregate_ensemble_attention
    
    def debug_aggregate(attention_list, aggregation_method, attention_layer, attention_head):
        print(f"\nDEBUG AGGREGATION:")
        print(f"  Input list length: {len(attention_list)}")
        print(f"  Aggregation method: {aggregation_method}")
        print(f"  Layer filter: {attention_layer}")
        print(f"  Head filter: {attention_head}")
        
        for i, att in enumerate(attention_list):
            if att is not None:
                print(f"  Item {i}: shape {att.shape}, type {type(att)}")
                if hasattr(att, 'unique'):
                    unique_vals = att.unique()
                    print(f"    Unique values: {len(unique_vals)} (range: {att.min():.6f} to {att.max():.6f})")
            else:
                print(f"  Item {i}: None")
        
        result = original_aggregate(attention_list, aggregation_method, attention_layer, attention_head)
        
        if result is not None:
            print(f"  Result shape: {result.shape}")
            if hasattr(result, 'unique'):
                unique_vals = result.unique()
                print(f"  Result unique values: {len(unique_vals)} (range: {result.min():.6f} to {result.max():.6f})")
        
        return result
    
    regressor._aggregate_ensemble_attention = debug_aggregate
    
    # Test aggregation
    pred, attention = regressor.predict(
        test_sample,
        return_attention=True,
        attention_type="features",
        attention_aggregation="mean"
    )
    
    # Restore original function
    regressor._aggregate_ensemble_attention = original_aggregate
    
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)
    
    print("Issues identified:")
    print("1. ❓ Identical values in aggregated attention - check aggregation function")
    print("2. ❓ Target variable involvement - check sequence structure")
    print("3. ❓ Empty attention patterns - check indexing and sequence alignment")
    print("4. ❓ Feature name alignment - check extended feature names")
    
    return attention_results

if __name__ == "__main__":
    results = debug_attention_issues()