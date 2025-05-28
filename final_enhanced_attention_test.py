#!/usr/bin/env python3
"""
Enhanced attention extraction test and visualization for TabPFN.
Tests the new attention_type parameter and creates comprehensive visualizations.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.tabpfn.regressor import TabPFNRegressor


def test_enhanced_attention_extraction():
    """Test the enhanced attention extraction functionality."""
    print("Loading California housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Use a subset for faster testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use smaller subset for demonstration
    X_train = X_train[:100]
    y_train = y_train[:100]
    X_test = X_test[:20]
    y_test = y_test[:20]
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Feature names: {feature_names}")
    
    # Initialize and fit the model
    print("\nInitializing TabPFN regressor...")
    model = TabPFNRegressor(
        n_estimators=1,
        device="cpu",
        random_state=42
    )
    
    print("Fitting model...")
    model.fit(X_train, y_train)
    
    # Test different attention extraction configurations
    test_configs = [
        {"attention_layer": None, "attention_type": "features", "name": "All layers - Features"},
        {"attention_layer": None, "attention_type": "items", "name": "All layers - Items"},
        {"attention_layer": 0, "attention_type": "features", "name": "Layer 0 - Features"},
        {"attention_layer": 0, "attention_type": "items", "name": "Layer 0 - Items"},
        {"attention_layer": 5, "attention_type": "features", "name": "Layer 5 - Features"},
        {"attention_layer": 5, "attention_type": "items", "name": "Layer 5 - Items"},
        {"attention_layer": 11, "attention_type": "features", "name": "Layer 11 - Features"},
        {"attention_layer": 11, "attention_type": "items", "name": "Layer 11 - Items"},
    ]
    
    results = {}
    
    print("\n" + "="*60)
    print("Testing Enhanced Attention Extraction")
    print("="*60)
    
    for config in test_configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 40)
        
        try:
            predictions, attention_matrices = model.predict(
                X_test,
                return_attention=True,
                attention_layer=config["attention_layer"],
                attention_type=config["attention_type"]
            )
            
            attention_matrix = attention_matrices[0]
            
            print(f"✓ Predictions shape: {predictions.shape}")
            print(f"✓ Attention matrix shape: {attention_matrix.shape}")
            print(f"✓ Attention matrix type: {type(attention_matrix)}")
            
            # Store results
            results[config['name']] = {
                'predictions': predictions,
                'attention': attention_matrix,
                'config': config
            }
            
        except Exception as e:
            print(f"✗ Error: {e}")
            results[config['name']] = {'error': str(e)}
    
    # Create visualizations
    print("\n" + "="*60)
    print("Creating Visualizations")
    print("="*60)
    
    # 1. Feature attention visualization
    feature_configs = [name for name in results.keys() if "Features" in name and 'error' not in results[name]]
    if feature_configs:
        create_feature_attention_visualization(results, feature_configs, feature_names)
    
    # 2. Item attention visualization
    item_configs = [name for name in results.keys() if "Items" in name and 'error' not in results[name]]
    if item_configs:
        create_item_attention_visualization(results, item_configs)
    
    # 3. Layer comparison
    create_layer_comparison_visualization(results)
    
    print("\n" + "="*60)
    print("Enhanced Attention Extraction Test Completed!")
    print("="*60)
    
    return results


def create_feature_attention_visualization(results, config_names, feature_names):
    """Create visualizations for feature attention matrices."""
    print("\nCreating feature attention visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, config_name in enumerate(config_names[:4]):
        if i >= 4:
            break
            
        attention_matrix = results[config_name]['attention']
        
        # Process feature attention matrix
        # Shape: [batch*seq, heads, features, 1] -> [features, features]
        if len(attention_matrix.shape) == 4:
            # Average over first dimension and squeeze last dimension
            feature_attention = attention_matrix.mean(dim=0).squeeze(-1)
        else:
            feature_attention = attention_matrix.squeeze()
        
        # Convert to numpy for easier handling
        feature_attention_np = feature_attention.detach().numpy()
        
        # Check if we have a valid 2D matrix
        if len(feature_attention_np.shape) == 2:
            # Create heatmap
            ax = axes[i]
            sns.heatmap(
                feature_attention_np,
                annot=True,
                fmt='.3f',
                cmap='viridis',
                ax=ax,
                cbar=True
            )
            ax.set_title(f'{config_name}\nShape: {feature_attention_np.shape}')
            
            # Set feature names if available and size matches
            if len(feature_names) == feature_attention_np.shape[0]:
                ax.set_xticklabels(feature_names, rotation=45)
                ax.set_yticklabels(feature_names, rotation=0)
        else:
            ax = axes[i]
            ax.text(0.5, 0.5, f'{config_name}\nShape: {feature_attention_np.shape}\n(Not square matrix)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    
    # Hide unused subplots
    for i in range(len(config_names), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('enhanced_feature_attention_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved enhanced_feature_attention_comparison.png")
    plt.close()


def create_item_attention_visualization(results, config_names):
    """Create visualizations for item attention matrices."""
    print("\nCreating item attention visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, config_name in enumerate(config_names[:4]):
        if i >= 4:
            break
            
        attention_matrix = results[config_name]['attention']
        
        # Process item attention matrix
        # Shape: [heads, test_items, train_items, 1] -> [test_items, train_items]
        if len(attention_matrix.shape) == 4:
            item_attention = attention_matrix.mean(dim=0).squeeze(-1)
        else:
            item_attention = attention_matrix.squeeze()
        
        # Create heatmap
        ax = axes[i]
        sns.heatmap(
            item_attention.detach().numpy(),
            cmap='viridis',
            ax=ax,
            cbar=True
        )
        ax.set_title(f'{config_name}\nShape: {item_attention.shape}')
        ax.set_xlabel('Training Items')
        ax.set_ylabel('Test Items')
    
    # Hide unused subplots
    for i in range(len(config_names), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('enhanced_item_attention_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved enhanced_item_attention_comparison.png")
    plt.close()


def create_layer_comparison_visualization(results):
    """Create a comparison of attention patterns across layers."""
    print("\nCreating layer comparison visualization...")
    
    # Extract attention statistics by layer
    layer_stats = {}
    
    for name, result in results.items():
        if 'error' in result:
            continue
            
        config = result['config']
        attention = result['attention']
        
        if config['attention_layer'] is not None:
            layer_idx = config['attention_layer']
            attention_type = config['attention_type']
            
            # Calculate statistics
            mean_attention = float(attention.mean())
            std_attention = float(attention.std())
            max_attention = float(attention.max())
            min_attention = float(attention.min())
            
            if layer_idx not in layer_stats:
                layer_stats[layer_idx] = {}
            
            layer_stats[layer_idx][attention_type] = {
                'mean': mean_attention,
                'std': std_attention,
                'max': max_attention,
                'min': min_attention
            }
    
    if layer_stats:
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        layers = sorted(layer_stats.keys())
        metrics = ['mean', 'std', 'max', 'min']
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            feature_values = []
            item_values = []
            
            for layer in layers:
                if 'features' in layer_stats[layer]:
                    feature_values.append(layer_stats[layer]['features'][metric])
                else:
                    feature_values.append(0)
                    
                if 'items' in layer_stats[layer]:
                    item_values.append(layer_stats[layer]['items'][metric])
                else:
                    item_values.append(0)
            
            ax.plot(layers, feature_values, 'o-', label='Feature Attention', linewidth=2)
            ax.plot(layers, item_values, 's-', label='Item Attention', linewidth=2)
            ax.set_xlabel('Layer Index')
            ax.set_ylabel(f'Attention {metric.capitalize()}')
            ax.set_title(f'Attention {metric.capitalize()} by Layer')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_layer_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved enhanced_layer_comparison.png")
        plt.close()


if __name__ == "__main__":
    results = test_enhanced_attention_extraction()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful_configs = [name for name, result in results.items() if 'error' not in result]
    failed_configs = [name for name, result in results.items() if 'error' in result]
    
    print(f"✓ Successful configurations: {len(successful_configs)}")
    for config in successful_configs:
        attention_shape = results[config]['attention'].shape
        print(f"  - {config}: {attention_shape}")
    
    if failed_configs:
        print(f"✗ Failed configurations: {len(failed_configs)}")
        for config in failed_configs:
            print(f"  - {config}: {results[config]['error']}")
    
    print(f"\n✓ Generated visualizations:")
    print(f"  - enhanced_feature_attention_comparison.png")
    print(f"  - enhanced_item_attention_comparison.png") 
    print(f"  - enhanced_layer_comparison.png")