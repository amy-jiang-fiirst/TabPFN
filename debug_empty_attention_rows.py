#!/usr/bin/env python3
"""
Debug script to investigate why certain rows in attention matrix appear empty.
Specifically investigating the issue where Latitude, Longitude, and Target rows are blank.
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

def debug_empty_attention_rows():
    """Debug why certain attention rows appear empty."""
    print("=" * 80)
    print("DEBUGGING EMPTY ATTENTION ROWS")
    print("=" * 80)
    
    # Use California housing dataset to match the user's example
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names
    
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
    
    print("\n" + "=" * 80)
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
    print(f"Attention tensor type: {type(attention)}")
    
    # Find the test sample in the attention matrix
    test_start_idx = X_train.shape[0]
    print(f"Test sample should be at index: {test_start_idx}")
    
    if test_start_idx < attention.shape[0]:
        # Extract attention for the test sample
        test_attention = attention[test_start_idx, 0, :, 0]  # [seq_pos, head, feature, token]
        
        print(f"\nTest sample attention shape: {test_attention.shape}")
        print(f"Test sample attention type: {type(test_attention)}")
        
        # Convert to numpy for analysis
        att_numpy = test_attention.detach().numpy()
        
        print(f"\nAttention values analysis:")
        print(f"  Min: {att_numpy.min():.8f}")
        print(f"  Max: {att_numpy.max():.8f}")
        print(f"  Mean: {att_numpy.mean():.8f}")
        print(f"  Std: {att_numpy.std():.8f}")
        print(f"  Sum: {att_numpy.sum():.8f}")
        print(f"  Non-zero count: {np.count_nonzero(att_numpy)}/{len(att_numpy)}")
        
        # Check each feature individually
        n_input_features = X.shape[1]
        total_features = att_numpy.shape[0]
        n_extra_features = total_features - n_input_features
        
        print(f"\nFeature-by-feature analysis:")
        print(f"  Input features: {n_input_features}")
        print(f"  Total features: {total_features}")
        print(f"  Extra features: {n_extra_features}")
        
        # Create extended feature names
        extended_feature_names = list(feature_names) + [f"Extra_{i}" for i in range(n_extra_features)]
        if n_extra_features > 0:
            extended_feature_names[-n_extra_features] = "Target"
            if n_extra_features > 1:
                for i in range(1, n_extra_features):
                    extended_feature_names[-(n_extra_features-i)] = f"Special_Token_{i}"
        
        print(f"\nIndividual feature attention values:")
        for i, (name, val) in enumerate(zip(extended_feature_names, att_numpy)):
            print(f"  {i:2d}. {name:15s}: {val:.8f}")
            
        # Check if any values are exactly zero or very small
        zero_threshold = 1e-10
        very_small_threshold = 1e-6
        
        zero_indices = np.where(np.abs(att_numpy) < zero_threshold)[0]
        very_small_indices = np.where((np.abs(att_numpy) < very_small_threshold) & (np.abs(att_numpy) >= zero_threshold))[0]
        
        print(f"\nZero/small value analysis:")
        print(f"  Exactly zero (< {zero_threshold}): {len(zero_indices)} features")
        if len(zero_indices) > 0:
            print(f"    Indices: {zero_indices.tolist()}")
            print(f"    Features: {[extended_feature_names[i] for i in zero_indices]}")
            
        print(f"  Very small (< {very_small_threshold}): {len(very_small_indices)} features")
        if len(very_small_indices) > 0:
            print(f"    Indices: {very_small_indices.tolist()}")
            print(f"    Features: {[extended_feature_names[i] for i in very_small_indices]}")
        
        # Create visualization to see the issue
        print(f"\n" + "=" * 80)
        print("CREATING VISUALIZATION")
        print("=" * 80)
        
        # Create a matrix where each row is the same attention vector (to simulate the heatmap)
        matrix = np.tile(att_numpy, (len(extended_feature_names), 1))
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            xticklabels=extended_feature_names,
            yticklabels=extended_feature_names,
            cmap='viridis'
        )
        
        plt.title('Feature-wise Attention Map\nLayer 11, Head 0 (Debug)')
        plt.xlabel('Features (To)')
        plt.ylabel('Features (From)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('/workspace/TabPFN/debug_empty_rows_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Debug visualization saved: debug_empty_rows_visualization.png")
        
        # Now let's check the raw attention matrix structure
        print(f"\n" + "=" * 80)
        print("RAW ATTENTION MATRIX ANALYSIS")
        print("=" * 80)
        
        # Get the full attention matrix for all sequence positions
        full_attention = attention[:, 0, :, 0].detach().numpy()  # [seq, feature]
        
        print(f"Full attention matrix shape: {full_attention.shape}")
        print(f"Sequence length: {full_attention.shape[0]}")
        print(f"Feature dimension: {full_attention.shape[1]}")
        
        # Check each sequence position
        print(f"\nAttention by sequence position:")
        for seq_idx in range(min(5, full_attention.shape[0])):  # Show first 5
            seq_attention = full_attention[seq_idx]
            non_zero_count = np.count_nonzero(seq_attention)
            print(f"  Seq {seq_idx:2d}: non-zero={non_zero_count:2d}/{len(seq_attention)}, "
                  f"sum={seq_attention.sum():.6f}, "
                  f"range=[{seq_attention.min():.6f}, {seq_attention.max():.6f}]")
        
        if full_attention.shape[0] > 5:
            print(f"  ... (showing first 5 of {full_attention.shape[0]} sequence positions)")
            
        # Check the test sample specifically
        if test_start_idx < full_attention.shape[0]:
            test_seq_attention = full_attention[test_start_idx]
            print(f"\nTest sample (seq {test_start_idx}):")
            print(f"  Non-zero: {np.count_nonzero(test_seq_attention)}/{len(test_seq_attention)}")
            print(f"  Sum: {test_seq_attention.sum():.6f}")
            print(f"  Range: [{test_seq_attention.min():.6f}, {test_seq_attention.max():.6f}]")
            
            # Check if this matches what we extracted before
            matches_previous = np.allclose(test_seq_attention, att_numpy)
            print(f"  Matches previous extraction: {matches_previous}")
            
        # Let's also check different attention types
        print(f"\n" + "=" * 80)
        print("CHECKING DIFFERENT ATTENTION TYPES")
        print("=" * 80)
        
        attention_types = ["features", "items"]
        for att_type in attention_types:
            try:
                pred_type, attention_type = regressor.predict(
                    test_sample,
                    return_attention=True,
                    attention_type=att_type,
                    attention_layer=11,
                    attention_head=0
                )
                
                print(f"\nAttention type '{att_type}':")
                print(f"  Shape: {attention_type.shape}")
                
                if test_start_idx < attention_type.shape[0]:
                    test_att_type = attention_type[test_start_idx, 0, :, 0].detach().numpy()
                    non_zero_count = np.count_nonzero(test_att_type)
                    print(f"  Non-zero: {non_zero_count}/{len(test_att_type)}")
                    print(f"  Sum: {test_att_type.sum():.6f}")
                    print(f"  Range: [{test_att_type.min():.6f}, {test_att_type.max():.6f}]")
                    
            except Exception as e:
                print(f"  Error with attention type '{att_type}': {e}")
    
    else:
        print(f"❌ Test sample index {test_start_idx} is out of bounds for attention shape {attention.shape}")

if __name__ == "__main__":
    debug_empty_attention_rows()