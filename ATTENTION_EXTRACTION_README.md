# TabPFN Attention Extraction Implementation

## Overview

This implementation adds attention weight extraction functionality to TabPFN, allowing users to visualize and analyze how the model attends to different features during regression tasks.

## Key Features

- **Attention Weight Extraction**: Extract attention matrices from TabPFN during prediction
- **Feature-to-Feature Attention Visualization**: Generate heatmaps showing attention between features
- **Feature Importance Analysis**: Analyze feature importance based on attention weights
- **California Housing Dataset Example**: Complete example using sklearn's California housing dataset

## Implementation Details

### Modified Files

1. **`src/tabpfn/model/multi_head_attention.py`**
   - Added `_compute_with_attention()` method to bypass memory caching when extracting attention
   - Modified `compute_attention_heads()` to force manual attention computation when `return_attention=True`
   - Enhanced `forward()` method to handle attention extraction

2. **`src/tabpfn/model/layer.py`**
   - Updated `PerFeatureEncoderLayer.forward()` to propagate `return_attention` parameter
   - Modified attention mechanism calls to handle tuple returns when extracting attention
   - Added logic to collect and return attention probabilities from sublayers

3. **`src/tabpfn/model/transformer.py`**
   - Updated `PerFeatureTransformer._forward()` to accept and handle `return_attention` parameter
   - Modified transformer encoder/decoder calls to pass through attention extraction flag

4. **`src/tabpfn/inference.py`**
   - Updated all inference engine implementations to support `return_attention` parameter
   - Modified `iter_outputs()` methods in `InferenceEngineOnDemand`, `InferenceEngineCachePreprocessing`, and `InferenceEngineCacheKV`

5. **`src/tabpfn/regressor.py`**
   - Enhanced `TabPFNRegressor.predict()` to pass `return_attention` parameter to inference engines

### New Files

1. **`final_attention_visualization.py`**
   - Complete example script demonstrating attention extraction and visualization
   - Includes feature importance analysis and heatmap generation
   - Uses California housing dataset for regression testing

2. **`test_attention_extraction.py`**
   - Development/testing script with detailed debugging output
   - Useful for understanding the attention extraction pipeline

## Usage Example

```python
from tabpfn import TabPFNRegressor
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = fetch_california_housing()
X, y = data.data[:100], data.target[:100]  # Use subset for faster computation

# Train TabPFN
regressor = TabPFNRegressor(n_estimators=1, device='cpu')
regressor.fit(X[:70], y[:70])

# Extract attention weights during prediction
result = regressor.predict(X[70:], return_attention=True)
predictions, attention_probs = result

# Visualize attention matrix
attention_matrix = attention_probs[0]  # Get first layer attention
# Create heatmap visualization (see final_attention_visualization.py for complete code)
```

## Attention Matrix Structure

The extracted attention matrices have the following structure:
- **Shape**: `[num_heads, test_samples, train_samples, num_features]`
- **Interpretation**: Shows how much attention each feature pays to other features
- **Aggregation**: Can be averaged across heads and samples for feature-to-feature attention

## Visualization Outputs

The implementation generates several visualization files:

1. **`california_housing_layer_*_heatmap.png`**: Feature-to-feature attention heatmaps
2. **`feature_importance_attention.png`**: Bar chart showing feature importance based on attention weights

## Technical Notes

### Memory Caching Bypass

The implementation includes a special `_compute_with_attention()` method that bypasses the memory caching decorator when extracting attention weights. This is necessary because the caching mechanism doesn't handle tuple returns (output + attention weights).

### Manual Attention Computation

When `return_attention=True`, the system forces manual attention computation instead of using PyTorch's optimized `scaled_dot_product_attention`, as the latter doesn't reliably return attention weights in all configurations.

### Attention Propagation

The attention extraction parameter is propagated through the entire model pipeline:
1. `TabPFNRegressor.predict(return_attention=True)`
2. → `InferenceEngine.iter_outputs(return_attention=True)`
3. → `PerFeatureTransformer.forward(return_attention=True)`
4. → `PerFeatureEncoderLayer.forward(return_attention=True)`
5. → `MultiHeadAttention.forward(return_attention=True)`

## Performance Considerations

- Attention extraction adds computational overhead due to manual attention computation
- Memory usage increases when storing attention matrices
- For large datasets, consider using smaller subsets for attention analysis

## Future Enhancements

Potential improvements for future versions:
1. Layer-wise attention extraction (currently returns final layer only)
2. Attention visualization for classification tasks
3. Interactive attention exploration tools
4. Attention-based feature selection methods

## Dependencies

- PyTorch >= 2.0.0
- matplotlib
- seaborn
- scikit-learn
- numpy

## Authors

Implementation by OpenHands AI Assistant (2025-05-28)
Based on TabPFN architecture by the original TabPFN team