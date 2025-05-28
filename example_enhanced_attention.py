#!/usr/bin/env python3
"""
Simple example demonstrating enhanced attention extraction in TabPFN.
This script shows how to use the new attention_type and attention_layer parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.tabpfn.regressor import TabPFNRegressor


def main():
    """Demonstrate enhanced attention extraction functionality."""
    print("Enhanced TabPFN Attention Extraction Example")
    print("=" * 50)
    
    # Load and prepare data
    print("\n1. Loading California housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Use subset for faster demonstration
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_test = X_train[:50], X_test[:10]
    y_train, y_test = y_train[:50], y_test[:10]
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Features: {feature_names}")
    
    # Initialize and fit model
    print("\n2. Training TabPFN model...")
    model = TabPFNRegressor(n_estimators=1, device="cpu", random_state=42)
    model.fit(X_train, y_train)
    
    # Example 1: Feature attention from layer 0
    print("\n3. Extracting feature attention from layer 0...")
    predictions, feature_attention = model.predict(
        X_test,
        return_attention=True,
        attention_layer=0,
        attention_type="features"
    )
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Feature attention shape: {feature_attention[0].shape}")
    
    # Process and visualize feature attention
    feature_attn_matrix = feature_attention[0].mean(dim=0).squeeze().detach().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        feature_attn_matrix,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        xticklabels=feature_names[:feature_attn_matrix.shape[1]],
        yticklabels=feature_names[:feature_attn_matrix.shape[0]]
    )
    plt.title('Feature-to-Feature Attention (Layer 0)')
    plt.tight_layout()
    plt.savefig('example_feature_attention.png', dpi=150, bbox_inches='tight')
    print("✓ Saved example_feature_attention.png")
    plt.close()
    
    # Example 2: Item attention from layer 0
    print("\n4. Extracting item attention from layer 0...")
    predictions, item_attention = model.predict(
        X_test,
        return_attention=True,
        attention_layer=0,
        attention_type="items"
    )
    
    print(f"Item attention shape: {item_attention[0].shape}")
    
    # Process and visualize item attention
    item_attn_matrix = item_attention[0].mean(dim=0).squeeze().detach().numpy()
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        item_attn_matrix,
        cmap='viridis',
        cbar=True
    )
    plt.title('Item-to-Item Attention (Layer 0)')
    plt.xlabel('Training Items')
    plt.ylabel('Test Items')
    plt.tight_layout()
    plt.savefig('example_item_attention.png', dpi=150, bbox_inches='tight')
    print("✓ Saved example_item_attention.png")
    plt.close()
    
    # Example 3: Compare attention across layers
    print("\n5. Comparing attention across layers...")
    layers_to_compare = [0, 5, 11]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, layer in enumerate(layers_to_compare):
        _, attention = model.predict(
            X_test,
            return_attention=True,
            attention_layer=layer,
            attention_type="features"
        )
        
        attn_matrix = attention[0].mean(dim=0).squeeze().detach().numpy()
        
        sns.heatmap(
            attn_matrix,
            annot=True,
            fmt='.2f',
            cmap='viridis',
            ax=axes[i],
            cbar=True
        )
        axes[i].set_title(f'Layer {layer}')
        
        print(f"Layer {layer} - Mean attention: {attn_matrix.mean():.4f}")
    
    plt.tight_layout()
    plt.savefig('example_layer_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved example_layer_comparison.png")
    plt.close()
    
    # Example 4: Attention statistics
    print("\n6. Attention statistics summary...")
    
    # Feature attention statistics
    feature_stats = {
        'mean': float(feature_attn_matrix.mean()),
        'std': float(feature_attn_matrix.std()),
        'max': float(feature_attn_matrix.max()),
        'min': float(feature_attn_matrix.min())
    }
    
    # Item attention statistics  
    item_stats = {
        'mean': float(item_attn_matrix.mean()),
        'std': float(item_attn_matrix.std()),
        'max': float(item_attn_matrix.max()),
        'min': float(item_attn_matrix.min())
    }
    
    print("\nFeature Attention Statistics:")
    for stat, value in feature_stats.items():
        print(f"  {stat}: {value:.4f}")
    
    print("\nItem Attention Statistics:")
    for stat, value in item_stats.items():
        print(f"  {stat}: {value:.4f}")
    
    print("\n" + "=" * 50)
    print("Enhanced attention extraction example completed!")
    print("Generated files:")
    print("  - example_feature_attention.png")
    print("  - example_item_attention.png") 
    print("  - example_layer_comparison.png")
    print("=" * 50)


if __name__ == "__main__":
    main()