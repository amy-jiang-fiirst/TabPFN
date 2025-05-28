#!/usr/bin/env python3
"""
Fix the visualization bug where certain rows appear empty in attention heatmaps.
The attention extraction is working correctly, but the visualization has issues.
"""

import sys
import os
sys.path.insert(0, '/workspace/TabPFN/src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import torch

def create_proper_attention_visualization():
    """Create a proper attention visualization that shows all values correctly."""
    print("=" * 80)
    print("FIXING ATTENTION VISUALIZATION BUG")
    print("=" * 80)
    
    # Use California housing dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = list(housing.feature_names)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Feature names: {feature_names}")
    
    # Take a smaller subset for faster processing
    X_subset = X[:100]
    y_subset = y[:100]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset, test_size=0.3, random_state=42
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Create regressor
    regressor = TabPFNRegressor(device='cpu')
    regressor.fit(X_train, y_train)
    
    # Get attention for a single test sample
    test_sample = X_test[:1]
    
    print(f"\n" + "=" * 80)
    print("EXTRACTING ATTENTION - LAYER 11, HEAD 0")
    print("=" * 80)
    
    pred, attention = regressor.predict(
        test_sample,
        return_attention=True,
        attention_type="features",
        attention_layer=11,
        attention_head=0
    )
    
    print(f"Attention tensor shape: {attention.shape}")
    
    # Extract attention for the test sample
    test_start_idx = X_train.shape[0]
    test_attention = attention[test_start_idx, 0, :, 0].detach().numpy()
    
    print(f"Test sample attention shape: {test_attention.shape}")
    print(f"Test sample attention values: {test_attention}")
    
    # Create extended feature names (including target and special tokens)
    n_input_features = X.shape[1]
    total_features = test_attention.shape[0]
    n_extra_features = total_features - n_input_features
    
    extended_feature_names = feature_names.copy()
    if n_extra_features > 0:
        extended_feature_names.append("Target")
        for i in range(1, n_extra_features):
            extended_feature_names.append(f"Special_Token_{i}")
    
    print(f"\nExtended feature names: {extended_feature_names}")
    print(f"Number of features: {len(extended_feature_names)}")
    print(f"Attention values length: {len(test_attention)}")
    
    # Verify all values are non-zero
    print(f"\nAttention value analysis:")
    for i, (name, val) in enumerate(zip(extended_feature_names, test_attention)):
        print(f"  {i:2d}. {name:15s}: {val:.8f}")
    
    # Create multiple visualization approaches to identify the bug
    
    # 1. Simple bar plot to verify values
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(test_attention)), test_attention)
    plt.title('Attention Values - Bar Plot')
    plt.xlabel('Feature Index')
    plt.ylabel('Attention Value')
    plt.xticks(range(len(extended_feature_names)), extended_feature_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, test_attention)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Heatmap - Single row (correct approach)
    plt.subplot(2, 2, 2)
    attention_matrix_single = test_attention.reshape(1, -1)
    sns.heatmap(
        attention_matrix_single,
        annot=True,
        fmt='.3f',
        xticklabels=extended_feature_names,
        yticklabels=['Test Sample'],
        cmap='viridis',
        cbar_kws={'label': 'Attention Value'}
    )
    plt.title('Attention Heatmap - Single Row (Correct)')
    plt.xticks(rotation=45, ha='right')
    
    # 3. Heatmap - Square matrix (WRONG approach that might cause the bug)
    plt.subplot(2, 2, 3)
    # This is a common mistake - creating a square matrix by tiling
    attention_matrix_square = np.tile(test_attention, (len(test_attention), 1))
    sns.heatmap(
        attention_matrix_square,
        annot=True,
        fmt='.3f',
        xticklabels=extended_feature_names,
        yticklabels=extended_feature_names,
        cmap='viridis',
        cbar_kws={'label': 'Attention Value'}
    )
    plt.title('Attention Heatmap - Square Matrix (Potentially Buggy)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 4. Feature-to-feature attention matrix (if available)
    plt.subplot(2, 2, 4)
    # Extract the full attention matrix for all sequence positions
    full_attention = attention[:, 0, :, 0].detach().numpy()
    
    # Create a feature-to-feature attention by averaging over sequence positions
    feature_to_feature = np.mean(full_attention, axis=0)
    feature_to_feature_matrix = feature_to_feature.reshape(1, -1)
    
    sns.heatmap(
        feature_to_feature_matrix,
        annot=True,
        fmt='.3f',
        xticklabels=extended_feature_names,
        yticklabels=['Avg Attention'],
        cmap='viridis',
        cbar_kws={'label': 'Attention Value'}
    )
    plt.title('Average Feature Attention Across All Positions')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('/workspace/TabPFN/fixed_attention_visualization_debug.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Fixed visualization saved: fixed_attention_visualization_debug.png")
    
    # Now let's test the specific issue with identical values but different colors
    print(f"\n" + "=" * 80)
    print("INVESTIGATING IDENTICAL VALUES WITH DIFFERENT COLORS")
    print("=" * 80)
    
    # Check if there are any identical values
    unique_values, counts = np.unique(test_attention, return_counts=True)
    identical_values = unique_values[counts > 1]
    
    print(f"Unique attention values: {len(unique_values)}")
    print(f"Identical values found: {len(identical_values)}")
    
    if len(identical_values) > 0:
        print(f"Identical values: {identical_values}")
        for val in identical_values:
            indices = np.where(np.isclose(test_attention, val, rtol=1e-9))[0]
            features = [extended_feature_names[i] for i in indices]
            print(f"  Value {val:.8f} appears in features: {features}")
    else:
        print("No identical values found - all attention values are unique")
    
    # Test with different aggregation methods to see if that causes issues
    print(f"\n" + "=" * 80)
    print("TESTING DIFFERENT AGGREGATION METHODS")
    print("=" * 80)
    
    aggregation_methods = ["mean", "max"]
    for agg_method in aggregation_methods:
        try:
            pred_agg, attention_agg = regressor.predict(
                test_sample,
                return_attention=True,
                attention_type="features",
                attention_layer=11,
                attention_aggregation=agg_method
            )
            
            test_attention_agg = attention_agg[test_start_idx, 0, :, 0].detach().numpy()
            
            print(f"\nAggregation method: {agg_method}")
            print(f"  Shape: {test_attention_agg.shape}")
            print(f"  Min: {test_attention_agg.min():.8f}")
            print(f"  Max: {test_attention_agg.max():.8f}")
            print(f"  Mean: {test_attention_agg.mean():.8f}")
            print(f"  Sum: {test_attention_agg.sum():.8f}")
            
            # Check for identical values in aggregated attention
            unique_agg, counts_agg = np.unique(test_attention_agg, return_counts=True)
            identical_agg = unique_agg[counts_agg > 1]
            
            if len(identical_agg) > 0:
                print(f"  ⚠️  Identical values found: {identical_agg}")
                for val in identical_agg:
                    indices = np.where(np.isclose(test_attention_agg, val, rtol=1e-9))[0]
                    features = [extended_feature_names[i] for i in indices]
                    print(f"    Value {val:.8f} in features: {features}")
            else:
                print(f"  ✅ All values unique")
                
        except Exception as e:
            print(f"  ❌ Error with {agg_method}: {e}")
    
    # Test with all heads to see if that causes uniform values
    print(f"\n" + "=" * 80)
    print("TESTING ALL HEADS AGGREGATION")
    print("=" * 80)
    
    try:
        pred_all, attention_all = regressor.predict(
            test_sample,
            return_attention=True,
            attention_type="features",
            attention_layer=11,
            attention_aggregation="mean"  # This should aggregate across all heads
        )
        
        test_attention_all = attention_all[test_start_idx, 0, :, 0].detach().numpy()
        
        print(f"All heads aggregation:")
        print(f"  Shape: {test_attention_all.shape}")
        print(f"  Min: {test_attention_all.min():.8f}")
        print(f"  Max: {test_attention_all.max():.8f}")
        print(f"  Mean: {test_attention_all.mean():.8f}")
        print(f"  Sum: {test_attention_all.sum():.8f}")
        
        # Check for the specific issue: identical values (like 0.167)
        unique_all, counts_all = np.unique(test_attention_all, return_counts=True)
        
        print(f"  Unique values: {len(unique_all)}")
        print(f"  Values: {unique_all}")
        
        # Check if we get the problematic 0.167 value
        if np.any(np.isclose(unique_all, 0.167, rtol=1e-3)):
            print(f"  ⚠️  Found 0.167-like values!")
            
        # Check if all values are identical (uniform distribution)
        if len(unique_all) == 1:
            print(f"  ⚠️  ALL VALUES ARE IDENTICAL: {unique_all[0]:.8f}")
            print(f"  This explains the user's issue!")
        elif np.allclose(test_attention_all, test_attention_all[0], rtol=1e-6):
            print(f"  ⚠️  ALL VALUES ARE NEARLY IDENTICAL")
            print(f"  This could cause visualization issues!")
            
        # Print individual values
        print(f"\nIndividual values for all-heads aggregation:")
        for i, (name, val) in enumerate(zip(extended_feature_names, test_attention_all)):
            print(f"  {i:2d}. {name:15s}: {val:.8f}")
            
    except Exception as e:
        print(f"❌ Error with all heads aggregation: {e}")

if __name__ == "__main__":
    create_proper_attention_visualization()