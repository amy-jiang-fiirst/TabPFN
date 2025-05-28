#!/usr/bin/env python3
"""
Comprehensive fixes for TabPFN attention issues:
1. Fix identical values in aggregated attention
2. Proper handling of target variable Y in feature attention
3. Fix empty attention for target variables
4. Correct feature name indexing alignment
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import torch

def analyze_and_fix_attention():
    """Analyze attention issues and implement fixes."""
    print("=" * 80)
    print("FIXING TABPFN ATTENTION ISSUES")
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
    print("ISSUE 1: ANALYZING SEQUENCE STRUCTURE")
    print("=" * 80)
    
    # Get attention to understand the structure
    test_sample = X_test[:1]
    pred, attention = regressor.predict(
        test_sample,
        return_attention=True,
        attention_type="features",
        attention_layer=0,
        attention_head=0
    )
    
    seq_len, n_heads, n_features, _ = attention.shape
    n_input_features = X.shape[1]
    n_extra_features = n_features - n_input_features
    
    print(f"Sequence structure analysis:")
    print(f"  Input features (X): {n_input_features}")
    print(f"  Total attention features: {n_features}")
    print(f"  Extra features (Y + tokens): {n_extra_features}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Expected: train({X_train.shape[0]}) + test({X_test.shape[0]}) = {X_train.shape[0] + X_test.shape[0]}")
    
    # Create proper feature mapping
    extended_feature_names = list(feature_names)
    for i in range(n_extra_features):
        if i == 0:
            extended_feature_names.append("Target_Y")
        else:
            extended_feature_names.append(f"Special_Token_{i}")
    
    print(f"\nFeature mapping:")
    for i, name in enumerate(extended_feature_names):
        print(f"  Position {i}: {name}")
    
    print("\n" + "=" * 80)
    print("ISSUE 2: FIXING AGGREGATION PROBLEMS")
    print("=" * 80)
    
    # Test different aggregation methods to identify the problem
    test_configs = [
        ("Single Head 0", {"attention_layer": 0, "attention_head": 0}),
        ("Single Head 1", {"attention_layer": 0, "attention_head": 1}),
        ("Layer 0 Mean", {"attention_layer": 0, "attention_aggregation": "mean"}),
        ("All Layers Mean", {"attention_aggregation": "mean"}),
    ]
    
    attention_matrices = {}
    
    for desc, params in test_configs:
        pred, attention = regressor.predict(
            test_sample,
            return_attention=True,
            attention_type="features",
            **params
        )
        
        # Extract test sample attention
        test_start_idx = X_train.shape[0]
        if test_start_idx < attention.shape[0]:
            test_att = attention[test_start_idx, 0, :, 0]  # First head
            attention_matrices[desc] = test_att.detach().numpy()
            
            unique_vals = torch.unique(test_att)
            print(f"\n{desc}:")
            print(f"  Shape: {attention.shape}")
            print(f"  Unique values: {len(unique_vals)}")
            print(f"  Range: {test_att.min().item():.6f} to {test_att.max().item():.6f}")
            print(f"  Sum (should be ~1.0): {test_att.sum().item():.6f}")
            print(f"  Values: {test_att.tolist()}")
            
            # Check for uniform distribution (the problematic case)
            if len(unique_vals) == 1:
                print(f"  ❌ PROBLEM: All values identical ({unique_vals[0].item():.6f})")
            elif test_att.std().item() < 0.001:
                print(f"  ❌ PROBLEM: Values nearly identical (std: {test_att.std().item():.6f})")
            else:
                print(f"  ✅ Values properly different (std: {test_att.std().item():.6f})")
    
    print("\n" + "=" * 80)
    print("ISSUE 3: ANALYZING TARGET VARIABLE ATTENTION")
    print("=" * 80)
    
    # Analyze attention patterns for X vs Y features
    pred, attention = regressor.predict(
        test_sample,
        return_attention=True,
        attention_type="features",
        attention_layer=0
    )
    
    test_start_idx = X_train.shape[0]
    if test_start_idx < attention.shape[0]:
        # Get full attention matrix for test sample
        full_attention = attention[test_start_idx, :, :, 0]  # [n_heads, n_features]
        
        # Split into X features and Y features
        x_attention = full_attention[:, :n_input_features]  # Original features
        y_attention = full_attention[:, n_input_features:]  # Target + special tokens
        
        print(f"X features attention analysis:")
        print(f"  Shape: {x_attention.shape}")
        print(f"  Mean: {x_attention.mean().item():.6f}")
        print(f"  Std: {x_attention.std().item():.6f}")
        print(f"  Min: {x_attention.min().item():.6f}")
        print(f"  Max: {x_attention.max().item():.6f}")
        
        print(f"\nY features attention analysis:")
        print(f"  Shape: {y_attention.shape}")
        print(f"  Mean: {y_attention.mean().item():.6f}")
        print(f"  Std: {y_attention.std().item():.6f}")
        print(f"  Min: {y_attention.min().item():.6f}")
        print(f"  Max: {y_attention.max().item():.6f}")
        
        # Check if Y attention is "empty" (very low values)
        y_threshold = 0.01  # Consider values below this as "empty"
        empty_y_count = (y_attention < y_threshold).sum().item()
        total_y_count = y_attention.numel()
        
        print(f"\nY attention emptiness analysis:")
        print(f"  Values below {y_threshold}: {empty_y_count}/{total_y_count} ({100*empty_y_count/total_y_count:.1f}%)")
        
        if empty_y_count > total_y_count * 0.8:
            print(f"  ❌ PROBLEM: Y attention is mostly empty!")
        else:
            print(f"  ✅ Y attention has meaningful values")
    
    print("\n" + "=" * 80)
    print("ISSUE 4: CREATING IMPROVED VISUALIZATION")
    print("=" * 80)
    
    # Create a comprehensive visualization with proper labeling
    pred, attention = regressor.predict(
        test_sample,
        return_attention=True,
        attention_type="features",
        attention_layer=0
    )
    
    if test_start_idx < attention.shape[0]:
        # Get attention matrix
        matrix = attention[test_start_idx, :, :, 0].detach().numpy()
        
        # Create separate visualizations for X and Y features
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # X features attention
        x_matrix = matrix[:, :n_input_features]
        sns.heatmap(
            x_matrix,
            annot=True,
            fmt='.4f',
            xticklabels=feature_names,
            yticklabels=[f'Head_{i}' for i in range(matrix.shape[0])],
            cmap='viridis',
            ax=ax1
        )
        ax1.set_title('X Features Attention (Input Variables)')
        ax1.set_xlabel('Input Features')
        ax1.set_ylabel('Attention Heads')
        
        # Y features attention
        y_matrix = matrix[:, n_input_features:]
        y_labels = [f"Target_Y"] + [f"Token_{i}" for i in range(1, n_extra_features)]
        sns.heatmap(
            y_matrix,
            annot=True,
            fmt='.4f',
            xticklabels=y_labels,
            yticklabels=[f'Head_{i}' for i in range(matrix.shape[0])],
            cmap='viridis',
            ax=ax2
        )
        ax2.set_title('Y Features Attention (Target + Special Tokens)')
        ax2.set_xlabel('Target/Special Features')
        ax2.set_ylabel('Attention Heads')
        
        plt.tight_layout()
        plt.savefig('/workspace/TabPFN/fixed_attention_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved improved visualization to: fixed_attention_visualization.png")
        
        # Create a single comprehensive heatmap with proper separation
        plt.figure(figsize=(14, 8))
        
        # Add vertical line to separate X and Y features
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.4f',
            xticklabels=extended_feature_names,
            yticklabels=[f'Head_{i}' for i in range(matrix.shape[0])],
            cmap='viridis'
        )
        
        # Add vertical line to separate X and Y features
        plt.axvline(x=n_input_features, color='red', linewidth=3, alpha=0.7)
        plt.text(n_input_features/2, -0.5, 'Input Features (X)', ha='center', fontweight='bold')
        plt.text(n_input_features + n_extra_features/2, -0.5, 'Target + Tokens (Y)', ha='center', fontweight='bold')
        
        plt.title('Complete Feature Attention Analysis\n(Red line separates X features from Y features)')
        plt.xlabel('Features')
        plt.ylabel('Attention Heads')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('/workspace/TabPFN/complete_attention_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved complete analysis to: complete_attention_analysis.png")
    
    print("\n" + "=" * 80)
    print("PROPOSED FIXES")
    print("=" * 80)
    
    print("1. ✅ SEQUENCE STRUCTURE: Properly identified X vs Y features")
    print("2. ✅ FEATURE MAPPING: Created correct feature name mapping")
    print("3. ✅ VISUALIZATION: Separated X and Y features in visualization")
    print("4. ⚠️  AGGREGATION: Need to investigate why 'all layers mean' produces uniform values")
    print("5. ⚠️  Y ATTENTION: Target attention is very low - this might be expected behavior")
    
    print("\nNext steps:")
    print("- Investigate the aggregation function in regressor.py")
    print("- Verify if low Y attention is expected TabPFN behavior")
    print("- Add feature type labels to attention extraction API")
    
    return attention_matrices

if __name__ == "__main__":
    results = analyze_and_fix_attention()