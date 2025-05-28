#!/usr/bin/env python3
"""
Comprehensive test to verify attention extraction fixes.
"""

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor

def test_prediction_consistency():
    """Test that predictions are consistent with and without attention extraction."""
    print("Testing prediction consistency...")
    
    # Generate test data
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create regressor
    regressor = TabPFNRegressor(device='cpu', )
    regressor.fit(X_train, y_train)
    
    # Get predictions without attention
    pred_normal = regressor.predict(X_test)
    
    # Get predictions with attention extraction (all layers)
    pred_with_attention, attention_all = regressor.predict(X_test, return_attention=True)
    
    # Get predictions with specific layer attention
    pred_layer_0, attention_layer_0 = regressor.predict(X_test, return_attention=True, attention_layer=0)
    
    # Get predictions with specific head attention
    pred_head_0, attention_head_0 = regressor.predict(X_test, return_attention=True, attention_head=0)
    
    # Check prediction consistency
    print(f"Normal predictions shape: {pred_normal.shape}")
    print(f"With attention predictions shape: {pred_with_attention.shape}")
    print(f"Layer 0 predictions shape: {pred_layer_0.shape}")
    print(f"Head 0 predictions shape: {pred_head_0.shape}")
    
    # Check if predictions are identical (within numerical precision)
    normal_vs_attention = np.allclose(pred_normal, pred_with_attention, rtol=1e-5, atol=1e-8)
    normal_vs_layer = np.allclose(pred_normal, pred_layer_0, rtol=1e-5, atol=1e-8)
    normal_vs_head = np.allclose(pred_normal, pred_head_0, rtol=1e-5, atol=1e-8)
    
    print(f"Normal vs With Attention consistent: {normal_vs_attention}")
    print(f"Normal vs Layer 0 consistent: {normal_vs_layer}")
    print(f"Normal vs Head 0 consistent: {normal_vs_head}")
    
    if not normal_vs_attention:
        print(f"Max difference (normal vs attention): {np.max(np.abs(pred_normal - pred_with_attention))}")
    if not normal_vs_layer:
        print(f"Max difference (normal vs layer 0): {np.max(np.abs(pred_normal - pred_layer_0))}")
    if not normal_vs_head:
        print(f"Max difference (normal vs head 0): {np.max(np.abs(pred_normal - pred_head_0))}")
    
    return normal_vs_attention and normal_vs_layer and normal_vs_head

def test_attention_shapes_and_values():
    """Test attention matrix shapes and value distributions."""
    print("\nTesting attention shapes and values...")
    
    # Generate test data
    X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create regressor
    regressor = TabPFNRegressor(device='cpu', )
    regressor.fit(X_train, y_train)
    
    # Test different attention extraction modes
    test_cases = [
        ("All layers", {}),
        ("Layer 0", {"attention_layer": 0}),
        ("Layer 1", {"attention_layer": 1}),
        ("Head 0", {"attention_head": 0}),
        ("Head 1", {"attention_head": 1}),
        ("Layer 0 Head 0", {"attention_layer": 0, "attention_head": 0}),
        ("Max aggregation", {"attention_aggregation": "max"}),
        ("Mean aggregation", {"attention_aggregation": "mean"}),
        ("Features attention", {"attention_type": "features"}),
        ("Items attention", {"attention_type": "items"}),
    ]
    
    results = {}
    
    for name, kwargs in test_cases:
        try:
            pred, attention = regressor.predict(X_test, return_attention=True, **kwargs)
            
            if attention is not None:
                if isinstance(attention, list):
                    print(f"\n{name}:")
                    print(f"  Attention is list with {len(attention)} elements")
                    for i, att in enumerate(attention):
                        if isinstance(att, torch.Tensor):
                            print(f"    Layer {i}: shape {att.shape}, min={att.min():.6f}, max={att.max():.6f}, mean={att.mean():.6f}")
                        else:
                            print(f"    Layer {i}: {type(att)}")
                elif isinstance(attention, torch.Tensor):
                    print(f"\n{name}:")
                    print(f"  Shape: {attention.shape}")
                    print(f"  Min: {attention.min():.6f}")
                    print(f"  Max: {attention.max():.6f}")
                    print(f"  Mean: {attention.mean():.6f}")
                    print(f"  Std: {attention.std():.6f}")
                    
                    # Check for uniform values (indicating aggregation issues)
                    unique_values = torch.unique(attention)
                    if len(unique_values) < 5:
                        print(f"  WARNING: Only {len(unique_values)} unique values found!")
                        print(f"  Unique values: {unique_values[:10]}")
                else:
                    print(f"\n{name}: Attention type: {type(attention)}")
                
                results[name] = attention
            else:
                print(f"\n{name}: No attention returned")
                results[name] = None
                
        except Exception as e:
            print(f"\n{name}: ERROR - {e}")
            results[name] = None
    
    return results

def test_layer_specific_extraction():
    """Test that layer-specific extraction works correctly."""
    print("\nTesting layer-specific extraction...")
    
    # Generate test data
    X, y = make_classification(n_samples=30, n_features=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create regressor
    regressor = TabPFNRegressor(device='cpu', )
    regressor.fit(X_train, y_train)
    
    # Get attention from all layers
    pred_all, attention_all = regressor.predict(X_test, return_attention=True)
    
    # Get attention from specific layers
    layer_attentions = {}
    for layer_idx in range(3):  # Test first 3 layers
        try:
            pred_layer, attention_layer = regressor.predict(X_test, return_attention=True, attention_layer=layer_idx)
            layer_attentions[layer_idx] = attention_layer
            print(f"Layer {layer_idx}: {type(attention_layer)}, shape: {attention_layer.shape if hasattr(attention_layer, 'shape') else 'N/A'}")
        except Exception as e:
            print(f"Layer {layer_idx}: ERROR - {e}")
            layer_attentions[layer_idx] = None
    
    # Check if layer-specific extraction returns different results
    if len(layer_attentions) >= 2:
        layer_0 = layer_attentions.get(0)
        layer_1 = layer_attentions.get(1)
        
        if layer_0 is not None and layer_1 is not None and hasattr(layer_0, 'shape') and hasattr(layer_1, 'shape'):
            if layer_0.shape == layer_1.shape:
                are_different = not torch.allclose(layer_0, layer_1, rtol=1e-5, atol=1e-8)
                print(f"Layer 0 vs Layer 1 are different: {are_different}")
                if not are_different:
                    print("WARNING: Layer-specific extraction may not be working correctly!")
                return are_different
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("TabPFN Attention Extraction Fix Verification")
    print("=" * 60)
    
    # Test 1: Prediction consistency
    consistency_ok = test_prediction_consistency()
    
    # Test 2: Attention shapes and values
    attention_results = test_attention_shapes_and_values()
    
    # Test 3: Layer-specific extraction
    layer_extraction_ok = test_layer_specific_extraction()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Prediction consistency: {'✓ PASS' if consistency_ok else '✗ FAIL'}")
    print(f"Layer-specific extraction: {'✓ PASS' if layer_extraction_ok else '✗ FAIL'}")
    
    # Count successful attention extractions
    successful_extractions = sum(1 for result in attention_results.values() if result is not None)
    total_extractions = len(attention_results)
    print(f"Attention extractions: {successful_extractions}/{total_extractions} successful")
    
    if consistency_ok and layer_extraction_ok and successful_extractions > total_extractions * 0.7:
        print("\n🎉 Most fixes appear to be working correctly!")
    else:
        print("\n⚠️  Some issues remain to be fixed.")

if __name__ == "__main__":
    main()