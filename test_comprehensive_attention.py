#!/usr/bin/env python3
"""
Comprehensive test of attention extraction functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor

def test_comprehensive_attention():
    """Test all attention extraction functionality."""
    print("Testing comprehensive attention extraction...")
    
    # Generate test data with meaningful feature names
    X, y = make_classification(
        n_samples=30, 
        n_features=4, 
        n_informative=3, 
        n_redundant=1, 
        n_classes=2, 
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    feature_names = ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D']
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Feature names: {feature_names}")
    
    # Create regressor
    regressor = TabPFNRegressor(device='cpu')
    regressor.fit(X_train, y_train)
    
    # Test 1: Prediction consistency
    print("\n=== Test 1: Prediction Consistency ===")
    pred_normal = regressor.predict(X_test)
    pred_with_attention, _ = regressor.predict(X_test, return_attention=True)
    
    print(f"Normal prediction shape: {pred_normal.shape}")
    print(f"Prediction with attention shape: {pred_with_attention.shape}")
    print(f"Predictions are identical: {np.allclose(pred_normal, pred_with_attention)}")
    
    # Test 2: Different attention types
    print("\n=== Test 2: Different Attention Types ===")
    
    for attention_type in ["features", "items"]:
        pred, attention = regressor.predict(
            X_test, 
            return_attention=True, 
            attention_layer=0,
            attention_type=attention_type
        )
        print(f"{attention_type.capitalize()} attention shape: {attention.shape}")
        
        # Check for empty attention (should not be empty)
        if attention is not None and hasattr(attention, 'numel'):
            non_zero_elements = (attention != 0).sum().item()
            total_elements = attention.numel()
            print(f"  Non-zero elements: {non_zero_elements}/{total_elements} ({100*non_zero_elements/total_elements:.1f}%)")
        
    # Test 3: Layer-specific extraction
    print("\n=== Test 3: Layer-specific Extraction ===")
    
    # Test different layers (TabPFN typically has 12 layers)
    for layer in [0, 5, 11]:
        try:
            pred, attention = regressor.predict(
                X_test, 
                return_attention=True, 
                attention_layer=layer,
                attention_type="features"
            )
            print(f"Layer {layer} attention shape: {attention.shape}")
        except Exception as e:
            print(f"Layer {layer} failed: {e}")
    
    # Test 4: Head-specific extraction
    print("\n=== Test 4: Head-specific Extraction ===")
    
    # Test different heads
    for head in [0, 1, 2]:
        try:
            pred, attention = regressor.predict(
                X_test, 
                return_attention=True, 
                attention_layer=0,
                attention_head=head,
                attention_type="features"
            )
            print(f"Head {head} attention shape: {attention.shape}")
            
            # Check some sample values to ensure they're different
            if attention is not None and attention.numel() > 0:
                sample_val = attention[0, 0, 0, 0].item()
                print(f"  Sample value [0,0,0,0]: {sample_val:.6f}")
                
        except Exception as e:
            print(f"Head {head} failed: {e}")
    
    # Test 5: Aggregation methods
    print("\n=== Test 5: Aggregation Methods ===")
    
    for aggregation in ["mean", "max"]:
        try:
            pred, attention = regressor.predict(
                X_test, 
                return_attention=True, 
                attention_layer=0,
                attention_aggregation=aggregation,
                attention_type="features"
            )
            print(f"{aggregation.capitalize()} aggregation attention shape: {attention.shape}")
            
            if attention is not None and attention.numel() > 0:
                sample_val = attention[0, 0, 0, 0].item()
                print(f"  Sample value [0,0,0,0]: {sample_val:.6f}")
                
        except Exception as e:
            print(f"{aggregation} aggregation failed: {e}")
    
    # Test 6: Visualization test
    print("\n=== Test 6: Basic Visualization Test ===")
    
    try:
        pred, attention = regressor.predict(
            X_test[:3],  # Use only first 3 test samples for visualization
            return_attention=True, 
            attention_layer=0,
            attention_type="features"
        )
        
        print(f"Visualization test attention shape: {attention.shape}")
        
        if attention is not None and attention.dim() >= 3:
            # Extract attention matrix - shape is [seq_len, n_heads, n_features, 1]
            print(f"Full attention shape: {attention.shape}")
            
            # For features attention, we want the attention from test samples to features
            # The attention matrix shows how much each position attends to each feature
            test_start_idx = X_train.shape[0]  # Start of test samples in sequence
            
            # Extract attention for first test sample (position test_start_idx)
            sample_attention = attention[test_start_idx, :, :, 0]  # Shape: [n_heads, n_features]
            print(f"Sample attention shape for visualization: {sample_attention.shape}")
            
            # Convert to numpy
            sample_attention_np = sample_attention.detach().numpy()
            
            # Create a simple heatmap
            plt.figure(figsize=(10, 6))
            
            # Create heatmap - each row is a head, each column is a feature
            sns.heatmap(
                sample_attention_np, 
                annot=True, 
                fmt='.3f',
                xticklabels=feature_names,
                yticklabels=[f'Head_{i}' for i in range(sample_attention_np.shape[0])],
                cmap='viridis'
            )
            plt.title('Feature Attention Heatmap (Layer 0, Test Sample 0)')
            plt.tight_layout()
            plt.savefig('/workspace/TabPFN/test_attention_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("Heatmap saved to test_attention_heatmap.png")
            
    except Exception as e:
        print(f"Visualization test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_comprehensive_attention()