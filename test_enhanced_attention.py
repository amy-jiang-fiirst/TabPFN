#!/usr/bin/env python3

"""
Test script for enhanced attention extraction functionality.
Tests layer-specific and head-specific attention extraction with aggregation methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor

def check_attention(attention, name):
    """Helper function to check attention matrices."""
    if attention is not None:
        if isinstance(attention, list):
            print(f"{name}: {len(attention)} matrices")
            # Find first non-None matrix
            for i, matrix in enumerate(attention):
                if matrix is not None:
                    print(f"Matrix {i} shape: {matrix.shape}")
                    print(f"Matrix {i} min/max: {matrix.min():.4f}/{matrix.max():.4f}")
                    return matrix  # Return first non-None matrix for visualization
            print("All matrices are None")
            return None
        else:
            print(f"{name} shape: {attention.shape}")
            print(f"{name} min/max: {attention.min():.4f}/{attention.max():.4f}")
            return attention
    else:
        print(f"No {name} extracted")
        return None

def test_enhanced_attention_extraction():
    """Test enhanced attention extraction with layer and head specific parameters."""
    
    print("Loading California housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names
    
    # Use a smaller subset for faster testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use even smaller subset for testing
    X_train = X_train[:100]
    y_train = y_train[:100]
    X_test = X_test[:20]
    y_test = y_test[:20]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    print(f"Feature names: {feature_names}")
    
    # Initialize TabPFN regressor
    regressor = TabPFNRegressor(device='cpu')
    
    # Fit the model
    print("Fitting TabPFN regressor...")
    regressor.fit(X_train_scaled, y_train)
    
    # Test 1: Extract attention from specific layer (layer 0)
    print("\nTest 1: Extracting attention from layer 0...")
    predictions_layer0, attention_layer0 = regressor.predict(
        X_test_scaled, 
        return_attention=True,
        attention_layer=0,
        attention_aggregation="mean"
    )
    
    attention_layer0_matrix = check_attention(attention_layer0, "Layer 0 attention")
    
    # Test 2: Extract attention from specific head (head 0) with mean aggregation
    print("\nTest 2: Extracting attention from head 0 with mean aggregation...")
    predictions_head0, attention_head0 = regressor.predict(
        X_test_scaled, 
        return_attention=True,
        attention_head=0,
        attention_aggregation="mean"
    )
    
    attention_head0_matrix = check_attention(attention_head0, "Head 0 attention")
    
    # Test 3: Extract attention with max aggregation
    print("\nTest 3: Extracting attention with max aggregation...")
    predictions_max, attention_max = regressor.predict(
        X_test_scaled, 
        return_attention=True,
        attention_aggregation="max"
    )
    
    attention_max_matrix = check_attention(attention_max, "Max aggregated attention")
    
    # Test 4: Extract attention from specific layer and head
    print("\nTest 4: Extracting attention from layer 0, head 1...")
    predictions_layer0_head1, attention_layer0_head1 = regressor.predict(
        X_test_scaled, 
        return_attention=True,
        attention_layer=0,
        attention_head=1,
        attention_aggregation="mean"
    )
    
    attention_layer0_head1_matrix = check_attention(attention_layer0_head1, "Layer 0, Head 1 attention")
    
    # Test 5: Compare different aggregation methods
    print("\nTest 5: Comparing mean vs max aggregation...")
    _, attention_mean = regressor.predict(
        X_test_scaled, 
        return_attention=True,
        attention_aggregation="mean"
    )
    
    _, attention_max_comp = regressor.predict(
        X_test_scaled, 
        return_attention=True,
        attention_aggregation="max"
    )
    
    attention_mean_matrix = check_attention(attention_mean, "Mean aggregation")
    attention_max_comp_matrix = check_attention(attention_max_comp, "Max aggregation")
    
    if attention_mean_matrix is not None and attention_max_comp_matrix is not None:
        # Create comparison visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mean aggregation heatmap
        sns.heatmap(
            attention_mean_matrix.squeeze(),
            annot=True,
            fmt='.3f',
            xticklabels=feature_names,
            yticklabels=feature_names,
            ax=axes[0],
            cmap='Blues'
        )
        axes[0].set_title('Mean Aggregation - Feature Attention')
        axes[0].set_xlabel('Features')
        axes[0].set_ylabel('Features')
        
        # Max aggregation heatmap
        sns.heatmap(
            attention_max_comp_matrix.squeeze(),
            annot=True,
            fmt='.3f',
            xticklabels=feature_names,
            yticklabels=feature_names,
            ax=axes[1],
            cmap='Reds'
        )
        axes[1].set_title('Max Aggregation - Feature Attention')
        axes[1].set_xlabel('Features')
        axes[1].set_ylabel('Features')
        
        plt.tight_layout()
        plt.savefig('enhanced_attention_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved comparison visualization as 'enhanced_attention_comparison.png'")
        plt.close()
    
    # Test 6: Verify predictions are consistent
    print("\nTest 6: Verifying prediction consistency...")
    predictions_normal = regressor.predict(X_test_scaled)
    
    print(f"Normal predictions shape: {predictions_normal.shape}")
    print(f"Layer 0 predictions shape: {predictions_layer0.shape}")
    print(f"Predictions match: {np.allclose(predictions_normal, predictions_layer0, rtol=1e-5)}")
    
    # Calculate MSE
    mse = np.mean((predictions_normal - y_test) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")
    
    print("\nEnhanced attention extraction tests completed successfully!")
    return True

if __name__ == "__main__":
    test_enhanced_attention_extraction()