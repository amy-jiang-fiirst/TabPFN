# TabPFN Enhanced Attention Visualization Guide

This guide provides comprehensive documentation for the enhanced attention visualization capabilities of TabPFN, including debugging fixes and complete visualization features.

## 🔧 Bug Fixes Applied

### Issue: `scaled_dot_product_attention` TypeError
**Problem**: PyTorch 2.7.0+ doesn't support the `need_weights=True` parameter in `torch.nn.functional.scaled_dot_product_attention`.

**Error Message**:
```
TypeError: scaled_dot_product_attention() got an unexpected keyword argument 'need_weights'
```

**Solution**: Modified `/src/tabpfn/model/multi_head_attention.py` to implement manual attention computation when attention weights are needed:

```python
if return_attention:
    # Manual attention computation to get attention weights
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    
    # Compute attention scores
    scale = 1.0 / math.sqrt(q_t.size(-1)) if softmax_scale is None else softmax_scale
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    
    # Apply softmax to get attention weights
    ps = torch.softmax(scores, dim=-1)
    
    # Apply dropout if specified
    if dropout_p > 0:
        ps = torch.dropout(ps, dropout_p, train=True)
    
    # Apply attention weights to values
    attention_head_outputs = torch.matmul(ps, v_t)
else:
    # Use optimized scaled_dot_product_attention when weights not needed
    attention_head_outputs = torch.nn.functional.scaled_dot_product_attention(...)
```

## 🎯 Enhanced Features

### 1. Layer-Specific Attention Extraction
Extract attention from specific transformer layers:

```python
# Extract attention from layer 5
attention_matrix = model.predict(
    X_test,
    return_attention=True,
    attention_layer=5,  # Specific layer (0-11)
    attention_type="features"
)

# Extract attention from all layers (default)
attention_matrix = model.predict(
    X_test,
    return_attention=True,
    attention_layer=None,  # All layers
    attention_type="features"
)
```

### 2. Head-Specific Attention Extraction
Extract attention from specific attention heads:

```python
# Extract attention from head 2
attention_matrix = model.predict(
    X_test,
    return_attention=True,
    attention_head=2,  # Specific head
    attention_type="features"
)

# Extract attention from all heads (default)
attention_matrix = model.predict(
    X_test,
    return_attention=True,
    attention_head=None,  # All heads
    attention_type="features"
)
```

### 3. Attention Aggregation Methods
Aggregate attention across multiple heads:

```python
# Mean aggregation (default)
attention_matrix = model.predict(
    X_test,
    return_attention=True,
    attention_aggregation="mean",
    attention_type="features"
)

# Max aggregation
attention_matrix = model.predict(
    X_test,
    return_attention=True,
    attention_aggregation="max",
    attention_type="features"
)
```

### 4. Feature-wise vs Sample-wise Attention
Extract different types of attention patterns:

```python
# Feature-wise attention (feature-to-feature relationships)
feature_attention = model.predict(
    X_test,
    return_attention=True,
    attention_type="features"
)

# Sample-wise attention (sample-to-sample relationships)
sample_attention = model.predict(
    X_test,
    return_attention=True,
    attention_type="items"
)
```

## 📊 Visualization Classes

### TabPFNAttentionVisualizer
Main class for creating comprehensive attention visualizations.

```python
from enhanced_attention_visualization import TabPFNAttentionVisualizer

# Initialize visualizer
visualizer = TabPFNAttentionVisualizer(
    model=fitted_model,
    feature_names=feature_names,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)
```

#### Key Methods:

1. **Feature Attention Visualization**
```python
fig = visualizer.visualize_feature_attention(
    attention_layer=5,           # Specific layer or None
    attention_head=2,            # Specific head or None
    attention_aggregation="mean", # "mean" or "max"
    save_path="feature_attention.png"
)
```

2. **Sample Attention Visualization**
```python
fig = visualizer.visualize_sample_attention(
    attention_layer=11,
    attention_head=None,
    attention_aggregation="max",
    save_path="sample_attention.png"
)
```

3. **Comprehensive Visualization**
```python
figures = visualizer.create_comprehensive_visualization(
    attention_layer=5,
    attention_head=0,
    attention_aggregation="mean",
    save_prefix="analysis"
)
# Creates: analysis_features.png, analysis_samples.png
```

4. **Layer Comparison**
```python
fig = visualizer.compare_layers(
    layers=[0, 5, 11],
    attention_type="features",
    save_path="layer_comparison.png"
)
```

5. **Head Comparison**
```python
fig = visualizer.compare_heads(
    heads=[0, 1, 2],
    attention_layer=5,
    attention_type="features",
    save_path="head_comparison.png"
)
```

## 🎨 Visualization Features

### Feature-wise Attention Maps
- **Rows/Columns**: Original feature names + "Target" variable
- **Values**: Attention weights between features
- **Interpretation**: How much each feature attends to other features

Example feature names displayed:
```
['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'Target']
```

### Sample-wise Attention Maps
- **Rows**: Test sample indices (`Test_0`, `Test_1`, ...)
- **Columns**: Training sample indices (`Train_0`, `Train_1`, ...)
- **Values**: Attention weights between samples
- **Interpretation**: How much each test sample attends to training samples

## 🚀 Quick Start Examples

### Example 1: Basic Usage
```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.tabpfn.regressor import TabPFNRegressor
from enhanced_attention_visualization import TabPFNAttentionVisualizer

# Load data
data = fetch_california_housing()
X, y = data.data[:200], data.target[:200]  # Subset for demo
feature_names = list(data.feature_names)

# Split and prepare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = TabPFNRegressor(device="cpu", random_state=42)
model.fit(X_train, y_train)

# Create visualizer
visualizer = TabPFNAttentionVisualizer(
    model, feature_names, X_train, X_test, y_train, y_test
)

# Generate comprehensive visualizations
visualizer.create_comprehensive_visualization(
    attention_layer=5,
    attention_head=None,
    attention_aggregation="mean",
    save_prefix="my_analysis"
)
```

### Example 2: Layer Comparison
```python
# Compare attention patterns across layers
visualizer.compare_layers(
    layers=[0, 5, 11],
    attention_type="features",
    save_path="feature_layer_comparison.png"
)

visualizer.compare_layers(
    layers=[0, 5, 11],
    attention_type="items",
    save_path="sample_layer_comparison.png"
)
```

### Example 3: Head Analysis
```python
# Analyze different attention heads in layer 5
visualizer.compare_heads(
    heads=[0, 1, 2, 3],
    attention_layer=5,
    attention_type="features",
    save_path="head_analysis.png"
)
```

### Example 4: Aggregation Comparison
```python
# Compare mean vs max aggregation
visualizer.create_comprehensive_visualization(
    attention_layer=11,
    attention_aggregation="mean",
    save_prefix="final_layer_mean"
)

visualizer.create_comprehensive_visualization(
    attention_layer=11,
    attention_aggregation="max",
    save_prefix="final_layer_max"
)
```

## 📁 File Structure

```
TabPFN/
├── enhanced_attention_visualization.py    # Main visualization class
├── interactive_attention_demo.py          # Interactive demo script
├── final_enhanced_attention_test.py       # Test script
├── src/tabpfn/model/multi_head_attention.py  # Fixed attention module
└── ATTENTION_VISUALIZATION_GUIDE.md       # This guide
```

## 🔍 Understanding Attention Shapes

### Feature Attention
- **Input Shape**: `[batch*seq, heads, features, 1]`
- **Processed Shape**: `[features, features]`
- **Interpretation**: `attention[i,j]` = how much feature `i` attends to feature `j`

### Sample Attention  
- **Input Shape**: `[heads, test_samples, train_samples, 1]`
- **Processed Shape**: `[test_samples, train_samples]`
- **Interpretation**: `attention[i,j]` = how much test sample `i` attends to train sample `j`

## ⚙️ Configuration Parameters

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `attention_layer` | `int` or `None` | `0-11` or `None` | Specific layer or all layers |
| `attention_head` | `int` or `None` | `0-N` or `None` | Specific head or all heads |
| `attention_aggregation` | `str` | `"mean"`, `"max"` | How to aggregate across heads |
| `attention_type` | `str` | `"features"`, `"items"` | Type of attention to extract |

## 🎯 Use Cases

1. **Feature Importance Analysis**: Use feature-wise attention to understand which features the model focuses on
2. **Sample Similarity**: Use sample-wise attention to see which training samples influence test predictions
3. **Layer Analysis**: Compare attention patterns across different transformer layers
4. **Head Specialization**: Analyze what different attention heads learn
5. **Model Interpretability**: Understand model decision-making process

## 🐛 Troubleshooting

### Common Issues:

1. **PyTorch Version Compatibility**: Ensure you're using the fixed `multi_head_attention.py`
2. **Memory Issues**: Use smaller datasets for visualization
3. **CUDA Errors**: Set `device="cpu"` for compatibility
4. **Shape Mismatches**: Check that your data preprocessing matches the expected format

### Debug Mode:
```python
# Enable verbose output
predictions, attention = model.predict(
    X_test,
    return_attention=True,
    attention_layer=5,
    attention_type="features"
)
print(f"Attention shape: {attention[0].shape}")
print(f"Attention type: {type(attention[0])}")
```

## 📈 Performance Tips

1. **Use CPU for small datasets**: `device="cpu"` is often faster for small datasets
2. **Subset large datasets**: Use smaller subsets for interactive exploration
3. **Cache results**: Save attention matrices for repeated visualization
4. **Batch processing**: Process multiple configurations in a single script

## 🔮 Future Enhancements

Potential improvements for future versions:
- Interactive web-based visualizations
- 3D attention visualizations
- Attention flow animations
- Statistical significance testing
- Attention pattern clustering
- Export to various formats (SVG, PDF, etc.)

---

For more examples and advanced usage, see the included demo scripts:
- `interactive_attention_demo.py`: Interactive demonstrations
- `final_enhanced_attention_test.py`: Comprehensive testing
- `enhanced_attention_visualization.py`: Full implementation