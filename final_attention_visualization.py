#!/usr/bin/env python3
"""
TabPFN Attention Extraction and Visualization
==============================================

This script demonstrates how to extract attention weights from TabPFN during regression
and visualize them as heatmaps showing feature interactions.

Author: OpenHands AI Assistant
Date: 2025-05-28
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor


def load_and_prepare_data():
    """Load and prepare the California housing dataset."""
    print("Loading California housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Feature names: {feature_names}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Use smaller subset for faster computation
    X_train = X_train[:70]  # TabPFN works well with smaller datasets
    y_train = y_train[:70]
    X_test = X_test[:30]
    y_test = y_test[:30]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


def extract_attention_weights(X_train, X_test, y_train):
    """Train TabPFN and extract attention weights during prediction."""
    print("Fitting TabPFN regressor...")
    regressor = TabPFNRegressor(
        n_estimators=1,  # Use single estimator for cleaner attention extraction
        device='cpu'
    )
    regressor.fit(X_train, y_train)
    
    print("Making predictions with attention extraction...")
    result = regressor.predict(X_test, return_attention=True)
    
    if isinstance(result, tuple):
        predictions, attention_probs = result
        print(f"Successfully extracted attention probabilities!")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Number of attention matrices: {len(attention_probs)}")
        
        for i, attn in enumerate(attention_probs):
            if attn is not None:
                print(f"Attention matrix {i} shape: {attn.shape}")
        
        return predictions, attention_probs
    else:
        print("Failed to extract attention probabilities")
        return result, None


def create_attention_heatmap(attention_matrix, feature_names, layer_idx=0, save_path=None):
    """Create and save attention heatmap visualization."""
    if attention_matrix is None:
        print("No attention matrix to visualize")
        return
    
    print(f"Creating attention heatmap for layer {layer_idx}...")
    
    # Convert to numpy if it's a tensor
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.detach().cpu().numpy()
    
    # Handle different attention matrix shapes
    # Expected shape: [batch_size, seq_len, seq_len, num_heads] or similar
    print(f"Attention matrix shape: {attention_matrix.shape}")
    
    # For TabPFN, we typically get shape [num_heads, test_samples, train_samples, num_features]
    # Let's average over heads and samples to get feature-to-feature attention
    if len(attention_matrix.shape) == 4:
        # Average over heads (first dimension) and test samples (second dimension)
        # This gives us [train_samples, num_features]
        avg_attention = np.mean(attention_matrix, axis=(0, 1))
        
        # If we have more train samples than features, we need to aggregate
        if avg_attention.shape[0] > len(feature_names):
            # Take the last num_features rows (corresponding to test samples)
            feature_attention = avg_attention[-len(feature_names):, :]
        else:
            feature_attention = avg_attention
    else:
        # Fallback: just use the matrix as-is if it's already 2D
        feature_attention = attention_matrix
    
    # Ensure we have the right shape for feature-to-feature attention
    if feature_attention.shape[0] != len(feature_names):
        print(f"Warning: Attention matrix shape {feature_attention.shape} doesn't match number of features {len(feature_names)}")
        # Try to reshape or slice to match
        min_dim = min(feature_attention.shape[0], len(feature_names))
        feature_attention = feature_attention[:min_dim, :min_dim]
        feature_names_subset = feature_names[:min_dim]
    else:
        feature_names_subset = feature_names
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        feature_attention,
        xticklabels=feature_names_subset,
        yticklabels=feature_names_subset,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": .8}
    )
    
    plt.title(f'TabPFN Attention Weights - Layer {layer_idx}\nFeature-to-Feature Interactions', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Features (Keys)', fontsize=12, fontweight='bold')
    plt.ylabel('Features (Queries)', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to: {save_path}")
    
    plt.show()
    return feature_attention


def analyze_feature_importance(attention_matrices, feature_names):
    """Analyze feature importance based on attention weights."""
    if not attention_matrices or attention_matrices[0] is None:
        print("No attention matrices available for analysis")
        return
    
    print("Analyzing feature importance from attention weights...")
    
    # Aggregate attention across all matrices
    total_attention = None
    for attn in attention_matrices:
        if attn is not None:
            if isinstance(attn, torch.Tensor):
                attn_np = attn.detach().cpu().numpy()
            else:
                attn_np = attn
            
            # Sum attention weights for each feature (sum over all dimensions except feature dim)
            feature_importance = np.sum(attn_np, axis=tuple(range(attn_np.ndim - 1)))
            
            if total_attention is None:
                total_attention = feature_importance
            else:
                total_attention += feature_importance
    
    if total_attention is not None:
        # Normalize to get relative importance
        total_attention = total_attention / np.sum(total_attention)
        
        # Create feature importance ranking
        importance_pairs = list(zip(feature_names, total_attention))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print("\nFeature Importance Ranking (based on attention weights):")
        for i, (feature, importance) in enumerate(importance_pairs, 1):
            print(f"{i:2d}. {feature:<15} : {importance:.4f}")
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        features, importances = zip(*importance_pairs)
        
        bars = plt.bar(range(len(features)), importances, color='skyblue', alpha=0.7)
        plt.xlabel('Features', fontweight='bold')
        plt.ylabel('Attention-based Importance', fontweight='bold')
        plt.title('Feature Importance from TabPFN Attention Weights', fontweight='bold')
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, importance in zip(bars, importances):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{importance:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('feature_importance_attention.png', dpi=300, bbox_inches='tight')
        print("Saved feature importance plot to: feature_importance_attention.png")
        plt.show()


def main():
    """Main function to run the complete attention extraction and visualization pipeline."""
    print("TabPFN Attention Extraction and Visualization")
    print("=" * 50)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    
    # Extract attention weights
    predictions, attention_probs = extract_attention_weights(X_train, X_test, y_train)
    
    if attention_probs is not None:
        print(f"\nPredictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        
        # Create attention heatmaps for each layer
        for i, attention_matrix in enumerate(attention_probs):
            if attention_matrix is not None:
                save_path = f"california_housing_layer_{i}_final_heatmap.png"
                create_attention_heatmap(attention_matrix, feature_names, i, save_path)
        
        # Analyze feature importance
        analyze_feature_importance(attention_probs, feature_names)
        
        print("\n" + "=" * 50)
        print("Attention extraction and visualization completed successfully!")
        print("Generated files:")
        print("- california_housing_layer_*_final_heatmap.png")
        print("- feature_importance_attention.png")
    else:
        print("Failed to extract attention probabilities")


if __name__ == "__main__":
    main()