# Enhanced Attention Extraction for TabPFN

This guide explains the enhanced attention extraction functionality implemented for TabPFN, which allows researchers to analyze attention patterns at different granularities.

## Overview

The enhanced attention extraction system provides:

1. **Layer-specific extraction**: Extract attention from specific transformer layers (0-11) or all layers
2. **Attention type differentiation**: Distinguish between feature-to-feature and item-to-item attention
3. **Head-specific support**: Framework for extracting attention from specific attention heads
4. **Comprehensive visualizations**: Generate heatmaps and comparative analyses

## API Reference

### TabPFNRegressor.predict() Enhanced Parameters

```python
def predict(
    self,
    X_test: np.ndarray,
    return_attention: bool = False,
    attention_layer: int | None = None,
    attention_head: int | None = None,
    attention_type: str = "features",
    attention_aggregation: str = "mean"
) -> tuple[np.ndarray, list[torch.Tensor]] | np.ndarray
```

#### Parameters

- **`return_attention`** (bool): Whether to return attention matrices along with predictions
- **`attention_layer`** (int | None): 
  - `None`: Extract attention from all layers (returns attention from the last layer)
  - `0-11`: Extract attention only from the specified layer
- **`attention_head`** (int | None): 
  - `None`: Aggregate attention across all heads
  - `0-5`: Extract attention from specific head (framework ready, not fully implemented)
- **`attention_type`** (str):
  - `"features"`: Extract feature-to-feature attention matrices
  - `"items"`: Extract item-to-item attention matrices
- **`attention_aggregation`** (str): How to aggregate multiple heads/layers (framework ready)

#### Returns

- **Without attention**: `np.ndarray` - Predictions only
- **With attention**: `tuple[np.ndarray, list[torch.Tensor]]` - Predictions and attention matrices

## Attention Matrix Shapes

### Feature Attention (`attention_type="features"`)

**Raw shape**: `[batch*sequence, heads, features, features, 1]`
**After aggregation**: `[features, features]` (typically `[6, 6]`)

This represents how features attend to each other during the prediction process.

### Item Attention (`attention_type="items"`)

**Raw shape**: `[heads, test_items, train_items, 1]`
**After aggregation**: `[test_items, train_items]`

This represents how test items attend to training items during prediction.

## Usage Examples

### Basic Attention Extraction

```python
from src.tabpfn.regressor import TabPFNRegressor
import numpy as np

# Initialize and fit model
model = TabPFNRegressor(device="cpu")
model.fit(X_train, y_train)

# Extract feature attention from all layers
predictions, attention = model.predict(
    X_test, 
    return_attention=True,
    attention_type="features"
)

print(f"Attention shape: {attention[0].shape}")
```

### Layer-Specific Extraction

```python
# Extract feature attention from layer 0 only
predictions, attention_layer0 = model.predict(
    X_test,
    return_attention=True,
    attention_layer=0,
    attention_type="features"
)

# Extract item attention from layer 11 only
predictions, attention_layer11 = model.predict(
    X_test,
    return_attention=True,
    attention_layer=11,
    attention_type="items"
)
```

### Comparative Analysis

```python
# Compare attention patterns across layers
layers_to_compare = [0, 5, 11]
attention_by_layer = {}

for layer in layers_to_compare:
    _, attention = model.predict(
        X_test,
        return_attention=True,
        attention_layer=layer,
        attention_type="features"
    )
    attention_by_layer[layer] = attention[0]

# Analyze differences
for layer, attn in attention_by_layer.items():
    print(f"Layer {layer} attention mean: {attn.mean():.4f}")
```

## Visualization Examples

### Feature Attention Heatmap

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Extract feature attention
predictions, attention = model.predict(
    X_test,
    return_attention=True,
    attention_layer=0,
    attention_type="features"
)

# Process attention matrix
feature_attention = attention[0].mean(dim=0).squeeze().detach().numpy()

# Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    feature_attention,
    annot=True,
    fmt='.3f',
    cmap='viridis',
    xticklabels=feature_names,
    yticklabels=feature_names
)
plt.title('Feature-to-Feature Attention (Layer 0)')
plt.tight_layout()
plt.savefig('feature_attention_heatmap.png')
```

### Item Attention Heatmap

```python
# Extract item attention
predictions, attention = model.predict(
    X_test,
    return_attention=True,
    attention_layer=0,
    attention_type="items"
)

# Process attention matrix
item_attention = attention[0].mean(dim=0).squeeze().detach().numpy()

# Create heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    item_attention,
    cmap='viridis',
    cbar=True
)
plt.title('Item-to-Item Attention (Layer 0)')
plt.xlabel('Training Items')
plt.ylabel('Test Items')
plt.tight_layout()
plt.savefig('item_attention_heatmap.png')
```

## Implementation Details

### Architecture Changes

1. **multi_head_attention.py**: Enhanced with layer/head-specific filtering
2. **layer.py**: Added attention type selection and conditional return logic
3. **transformer.py**: Fixed LayerStack to preserve non-None attention matrices
4. **inference.py**: Updated all InferenceEngine implementations
5. **regressor.py**: Enhanced public API with new parameters

### Key Bug Fixes

1. **LayerStack Attention Preservation**: Previously, attention from specific layers was being overwritten by `None` from subsequent layers
2. **Attention Type Selection**: Proper distinction between feature and item attention mechanisms
3. **Parameter Propagation**: Ensured all new parameters flow correctly through the entire call stack

## Testing

The implementation includes comprehensive tests in:

- `test_attention_type.py`: Basic functionality testing
- `test_enhanced_attention.py`: Helper functions for analysis
- `final_enhanced_attention_test.py`: Complete test suite with visualizations

### Test Coverage

- ✅ Layer-specific extraction (layers 0, 5, 11)
- ✅ Attention type differentiation (features vs items)
- ✅ Shape validation for all configurations
- ✅ Visualization generation
- ✅ Error handling and edge cases

## Performance Considerations

- **Memory Usage**: Attention matrices can be large for big datasets
- **Computation Time**: Extracting attention adds minimal overhead
- **Storage**: Consider saving attention matrices for large-scale analyses

## Research Applications

### Feature Interaction Analysis

Use feature attention to understand:
- Which features are most important for predictions
- How features interact with each other
- Feature dependency patterns across layers

### Training Data Influence

Use item attention to understand:
- Which training examples most influence each prediction
- How attention patterns change across layers
- Data efficiency and redundancy analysis

### Model Interpretability

- Compare attention patterns across different layers
- Analyze how attention evolves during the forward pass
- Identify potential biases or unexpected patterns

## Future Enhancements

The current implementation provides a foundation for:

1. **Head-specific analysis**: Extract attention from individual attention heads
2. **Temporal analysis**: Track attention changes across training epochs
3. **Comparative studies**: Compare attention patterns across different model configurations
4. **Advanced aggregation**: Implement sophisticated attention aggregation methods

## Troubleshooting

### Common Issues

1. **None attention matrices**: Ensure `return_attention=True` and valid layer indices
2. **Shape mismatches**: Check that attention_type matches your analysis needs
3. **Memory errors**: Use smaller batch sizes for large datasets

### Debug Tips

```python
# Check available layers
print(f"Model has {model.model.n_layers} layers")

# Verify attention extraction
predictions, attention = model.predict(X_test[:1], return_attention=True)
print(f"Attention shape: {attention[0].shape if attention[0] is not None else 'None'}")
```

## Conclusion

The enhanced attention extraction functionality provides researchers with powerful tools to analyze TabPFN's internal mechanisms. By supporting both feature-level and item-level attention analysis across different layers, this implementation enables deep insights into how TabPFN makes predictions and processes tabular data.