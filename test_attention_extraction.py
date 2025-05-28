#!/usr/bin/env python3
"""
Test script to verify attention extraction functionality and create enhanced visualizations
for TabPFN using the California housing dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor

def test_attention_extraction():
    """Test the existing attention extraction functionality."""
    print("Testing attention extraction functionality...")
    
    # Load California housing dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Feature names: {feature_names}")
    
    # Use a smaller subset for testing
    X_small = X[:100]  # Use first 100 samples
    y_small = y[:100]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_small, y_small, test_size=0.3, random_state=42
    )
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    # Initialize TabPFN regressor
    regressor = TabPFNRegressor(device='cpu', n_estimators=1)
    
    # Fit the model
    print("Fitting TabPFN regressor...")
    regressor.fit(X_train_scaled, y_train)
    
    # Make predictions with attention extraction
    print("Making predictions with attention extraction...")
    try:
        result = regressor.predict(X_test_scaled, return_attention=True)
        
        if isinstance(result, tuple) and len(result) == 2:
            predictions, attention_probs = result
            print(f"Predictions shape: {predictions.shape}")
            print(f"Attention probabilities type: {type(attention_probs)}")
            
            if attention_probs is not None:
                if isinstance(attention_probs, list):
                    print(f"Number of attention matrices: {len(attention_probs)}")
                    for i, attn in enumerate(attention_probs):
                        if attn is not None:
                            print(f"Attention matrix {i} shape: {attn.shape}")
                        else:
                            print(f"Attention matrix {i} is None")
                else:
                    print(f"Attention probabilities shape: {attention_probs.shape}")
            else:
                print("No attention probabilities returned")
                
            return predictions, attention_probs, feature_names, X_test_scaled
        else:
            print(f"Unexpected result format: {type(result)}")
            return None, None, feature_names, X_test_scaled
        
    except Exception as e:
        print(f"Error during prediction with attention: {e}")
        import traceback
        traceback.print_exc()
        return None, None, feature_names, X_test_scaled

def visualize_attention_heatmap(attention_probs, feature_names, save_path=None):
    """Create enhanced heatmap visualizations of attention matrices."""
    if attention_probs is None:
        print("No attention probabilities to visualize")
        return
    
    print("Creating attention heatmap visualizations...")
    
    # Handle different attention probability formats
    if isinstance(attention_probs, list):
        # Multiple layers
        for layer_idx, attn in enumerate(attention_probs):
            if attn is not None:
                _plot_single_attention_matrix(
                    attn, feature_names, f"Layer {layer_idx}", save_path
                )
    else:
        # Single attention matrix
        _plot_single_attention_matrix(
            attention_probs, feature_names, "Attention", save_path
        )

def _plot_single_attention_matrix(attention_matrix, feature_names, title_prefix, save_path=None):
    """Plot a single attention matrix as a heatmap."""
    # Convert to numpy if it's a tensor
    if torch.is_tensor(attention_matrix):
        attn_np = attention_matrix.detach().cpu().numpy()
    else:
        attn_np = np.array(attention_matrix)
    
    print(f"Plotting attention matrix with shape: {attn_np.shape}")
    
    # Handle different attention matrix shapes
    if len(attn_np.shape) == 4:  # (batch, seq_len, seq_len, heads)
        # Average over batch and heads
        attn_avg = np.mean(attn_np, axis=(0, -1))
    elif len(attn_np.shape) == 3:  # (batch, seq_len, seq_len) or (seq_len, seq_len, heads)
        if attn_np.shape[0] == attn_np.shape[1]:  # (seq_len, seq_len, heads)
            attn_avg = np.mean(attn_np, axis=-1)
        else:  # (batch, seq_len, seq_len)
            attn_avg = np.mean(attn_np, axis=0)
    elif len(attn_np.shape) == 2:  # (seq_len, seq_len)
        attn_avg = attn_np
    else:
        print(f"Unexpected attention matrix shape: {attn_np.shape}")
        return
    
    # Create feature labels (including target if present)
    if attn_avg.shape[0] == len(feature_names) + 1:
        labels = list(feature_names) + ['Target']
    else:
        labels = [f'Feature_{i}' for i in range(attn_avg.shape[0])]
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        attn_avg,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.title(f'{title_prefix} - Feature Attention Matrix\nCalifornia Housing Dataset')
    plt.xlabel('Key Features')
    plt.ylabel('Query Features')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        filename = f"{save_path}_{title_prefix.lower().replace(' ', '_')}_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to: {filename}")
    
    plt.show()

def analyze_feature_importance(attention_probs, feature_names):
    """Analyze feature importance based on attention weights."""
    if attention_probs is None:
        print("No attention probabilities to analyze")
        return
    
    print("Analyzing feature importance from attention weights...")
    
    # Convert to numpy if needed
    if torch.is_tensor(attention_probs):
        attn_np = attention_probs.detach().cpu().numpy()
    elif isinstance(attention_probs, list):
        # Use the last layer's attention if multiple layers
        attn_np = attention_probs[-1]
        if attn_np is None:
            print("Attention matrix is None - cannot analyze feature importance")
            return
        if torch.is_tensor(attn_np):
            attn_np = attn_np.detach().cpu().numpy()
    else:
        attn_np = np.array(attention_probs)
    
    # Average attention weights
    if len(attn_np.shape) == 4:  # (batch, seq_len, seq_len, heads)
        attn_avg = np.mean(attn_np, axis=(0, -1))
    elif len(attn_np.shape) == 3:
        if attn_np.shape[0] == attn_np.shape[1]:  # (seq_len, seq_len, heads)
            attn_avg = np.mean(attn_np, axis=-1)
        else:  # (batch, seq_len, seq_len)
            attn_avg = np.mean(attn_np, axis=0)
    else:
        attn_avg = attn_np
    
    # Calculate feature importance as sum of attention weights
    feature_importance = np.sum(attn_avg, axis=0)
    
    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    
    # Ensure we have the right number of features
    if len(feature_importance) == len(feature_names) + 1:
        labels = list(feature_names) + ['Target']
        importance = feature_importance
    else:
        labels = feature_names[:len(feature_importance)]
        importance = feature_importance[:len(feature_names)]
    
    # Sort by importance
    sorted_indices = np.argsort(importance)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_importance = importance[sorted_indices]
    
    # Create bar plot
    bars = plt.bar(range(len(sorted_importance)), sorted_importance, 
                   color='skyblue', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, sorted_importance)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Features')
    plt.ylabel('Attention Weight Sum')
    plt.title('Feature Importance Based on Attention Weights\nCalifornia Housing Dataset')
    plt.xticks(range(len(sorted_labels)), sorted_labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Print feature importance ranking
    print("\nFeature Importance Ranking (based on attention weights):")
    for i, (label, importance) in enumerate(zip(sorted_labels, sorted_importance)):
        print(f"{i+1:2d}. {label:20s}: {importance:.4f}")

def main():
    """Main function to run the attention extraction and visualization."""
    print("TabPFN Attention Extraction and Visualization")
    print("=" * 50)
    
    # Test attention extraction
    predictions, attention_probs, feature_names, X_test = test_attention_extraction()
    
    if predictions is not None:
        print(f"\nSuccessfully extracted attention probabilities!")
        print(f"Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        
        # Create visualizations
        visualize_attention_heatmap(attention_probs, feature_names, "california_housing")
        
        # Analyze feature importance
        analyze_feature_importance(attention_probs, feature_names)
        
    else:
        print("Failed to extract attention probabilities")

if __name__ == "__main__":
    main()