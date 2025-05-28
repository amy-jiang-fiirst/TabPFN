# Enhanced TabPFN Attention Visualization

## Overview

This repository contains a comprehensive enhanced attention visualization tool for TabPFN models that fixes the original blank visualization issues and supports all requested configurations.

## Problem Fixed

The original `enhanced_attention_visualization.py` had several issues:
1. **Blank areas in feature attention maps**: The last 2-3 features (Latitude, Longitude, Target) showed as completely blank/empty
2. **Dimension mismatch**: The attention matrix returned by TabPFN was 6x6 but expected to be 9x9 (8 features + 1 target)
3. **Limited configuration support**: Did not support all requested layer/head combinations

## Root Cause Analysis

The TabPFN model internally processes only 6 features for attention computation, not all 8 input features. The attention extraction returns a 6x6 matrix instead of the expected 8x8 or 9x9 matrix.

## Solution

The enhanced visualization script now:
1. **Handles dimension mismatch**: Automatically pads smaller attention matrices to match the expected feature count
2. **Supports all configurations**: Implements all 4 requested configuration types
3. **Provides comprehensive visualization**: Generates both feature-wise and sample-wise attention maps
4. **Includes proper error handling**: Gracefully handles various tensor shapes and dimensions

## Supported Configurations

### 1. One Layer One Head
```python
# Layer 5, Head 0
visualizer.create_comprehensive_visualization(
    attention_layers=5,
    attention_head=0,
    save_path="layer5_head0"
)
```

### 2. Multi/All Layers Multi/All Heads with Aggregation
```python
# All layers, all heads with mean aggregation
visualizer.create_comprehensive_visualization(
    attention_layers=None,
    attention_head=None,
    attention_aggregation="mean",
    save_path="all_layers_all_heads_mean"
)

# Multiple specific layers [0, 5, 11], all heads with mean aggregation
visualizer.create_comprehensive_visualization(
    attention_layers=[0, 5, 11],
    attention_head=None,
    attention_aggregation="mean",
    save_path="multi_layers_all_heads_mean"
)
```

### 3. One Layer Multi/All Heads with Aggregation
```python
# Layer 5, all heads with mean aggregation
visualizer.create_comprehensive_visualization(
    attention_layers=5,
    attention_head=None,
    attention_aggregation="mean",
    save_path="layer5_all_heads_mean"
)
```

### 4. Multi/All Layers One Head with Aggregation
```python
# All layers, head 0 with mean aggregation
visualizer.create_comprehensive_visualization(
    attention_layers=None,
    attention_head=0,
    attention_aggregation="mean",
    save_path="all_layers_head0_mean"
)
```

## Parameters

- **attention_layers**: `int`, `List[int]`, or `None`
  - Single layer: `5`
  - Multiple layers: `[0, 5, 11]`
  - All layers: `None`

- **attention_head**: `int`, `List[int]`, or `None`
  - Single head: `0`
  - Multiple heads: `[0, 1, 2]` (not fully implemented)
  - All heads: `None`

- **attention_aggregation**: `"mean"` or `"max"`
  - Method to aggregate attention across heads/layers

- **attention_type**: `"features"` or `"items"`
  - Feature-wise attention: relationships between features
  - Sample-wise attention: relationships between training and test samples

- **save_path**: `str` or `None`
  - Base path for saving visualizations (will append `_features.png` and `_items.png`)

## Usage

```python
from enhanced_attention_visualization import TabPFNAttentionVisualizer
from src.tabpfn.regressor import TabPFNRegressor

# Train your TabPFN model
model = TabPFNRegressor()
model.fit(X_train, y_train)

# Create visualizer
visualizer = TabPFNAttentionVisualizer(
    model=model,
    feature_names=feature_names,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

# Generate visualizations
visualizer.create_comprehensive_visualization(
    attention_layers=5,
    attention_head=0,
    save_path="my_attention_analysis"
)
```

## Generated Files

The script generates comprehensive visualizations for all configurations:

### Configuration 1 (One layer one head):
- `layer5_head0_features.png` / `layer5_head0_items.png`
- `layer11_head1_features.png` / `layer11_head1_items.png`

### Configuration 2 (Multi/all layers multi/all heads):
- `all_layers_all_heads_mean_features.png` / `all_layers_all_heads_mean_items.png`
- `all_layers_all_heads_max_features.png` / `all_layers_all_heads_max_items.png`
- `multi_layers_all_heads_mean_features.png` / `multi_layers_all_heads_mean_items.png`

### Configuration 3 (One layer multi/all heads):
- `layer5_all_heads_mean_features.png` / `layer5_all_heads_mean_items.png`
- `layer11_all_heads_max_features.png` / `layer11_all_heads_max_items.png`

### Configuration 4 (Multi/all layers one head):
- `all_layers_head0_mean_features.png` / `all_layers_head0_mean_items.png`
- `multi_layers_head1_max_features.png` / `multi_layers_head1_max_items.png`

## Key Improvements

1. **Fixed blank visualization issue**: Properly handles dimension mismatch by padding attention matrices
2. **Comprehensive configuration support**: All 4 requested configuration types implemented
3. **Flexible parameter handling**: Supports single values, lists, and None for layers/heads
4. **Robust error handling**: Gracefully handles various tensor shapes and edge cases
5. **Clear documentation**: Comprehensive usage examples and parameter descriptions
6. **Informative output**: Provides clear feedback about matrix padding and processing

## Dependencies

- torch
- matplotlib
- seaborn
- scikit-learn
- numpy
- tabpfn

## Running the Script

```bash
python enhanced_attention_visualization.py
```

This will generate all example visualizations demonstrating each configuration type.