#!/usr/bin/env python3
"""
Interactive demo for TabPFN attention visualization.

This script provides an easy-to-use interface for exploring TabPFN attention patterns
with different configurations.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Optional, Literal
import warnings
warnings.filterwarnings('ignore')

from src.tabpfn.regressor import TabPFNRegressor


def demo_attention_extraction():
    """Interactive demo for attention extraction with different parameters."""
    print("="*80)
    print("Interactive TabPFN Attention Visualization Demo")
    print("="*80)
    
    # Load and prepare data
    print("\nLoading California housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    
    # Split and prepare data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use smaller subset for faster demo
    X_train = X_train[:50]
    y_train = y_train[:50]
    X_test = X_test[:10]
    y_test = y_test[:10]
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Feature names: {feature_names}")
    
    # Initialize and fit model
    print("\nTraining TabPFN model...")
    model = TabPFNRegressor(
        n_estimators=1,
        device="cpu",
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Demo configurations
    demo_configs = [
        {
            "name": "All Layers, All Heads (Mean)",
            "attention_layer": None,
            "attention_head": None,
            "attention_aggregation": "mean",
            "description": "Aggregates attention across all layers and heads using mean"
        },
        {
            "name": "Layer 0, All Heads (Mean)",
            "attention_layer": 0,
            "attention_head": None,
            "attention_aggregation": "mean",
            "description": "First layer attention, averaged across all heads"
        },
        {
            "name": "Layer 5, All Heads (Max)",
            "attention_layer": 5,
            "attention_head": None,
            "attention_aggregation": "max",
            "description": "Middle layer attention, max aggregation across heads"
        },
        {
            "name": "Layer 11, All Heads (Mean)",
            "attention_layer": 11,
            "attention_head": None,
            "attention_aggregation": "mean",
            "description": "Final layer attention, averaged across all heads"
        },
        {
            "name": "Layer 5, Head 0",
            "attention_layer": 5,
            "attention_head": 0,
            "attention_aggregation": "mean",
            "description": "Specific attention head in middle layer"
        }
    ]
    
    print("\n" + "="*80)
    print("ATTENTION EXTRACTION DEMO")
    print("="*80)
    
    for i, config in enumerate(demo_configs, 1):
        print(f"\n{i}. {config['name']}")
        print(f"   Description: {config['description']}")
        print("-" * 60)
        
        # Extract and visualize feature attention
        print("   Extracting feature attention...")
        try:
            predictions, attention_matrices = model.predict(
                X_test,
                return_attention=True,
                attention_layer=config["attention_layer"],
                attention_head=config["attention_head"],
                attention_aggregation=config["attention_aggregation"],
                attention_type="features"
            )
            
            attention_matrix = attention_matrices[0]
            print(f"   ✓ Feature attention shape: {attention_matrix.shape}")
            
            # Process for visualization
            if len(attention_matrix.shape) == 4:
                feature_attention = attention_matrix.mean(dim=0).squeeze(-1)
                if len(feature_attention.shape) == 3:
                    feature_attention = feature_attention.mean(dim=0)
            else:
                feature_attention = attention_matrix.squeeze()
                
            feature_attention_np = feature_attention.detach().cpu().numpy()
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Feature attention heatmap
            extended_names = feature_names + ['Target']
            sns.heatmap(
                feature_attention_np,
                annot=True,
                fmt='.3f',
                cmap='viridis',
                ax=ax1,
                xticklabels=extended_names,
                yticklabels=extended_names
            )
            ax1.set_title(f'Feature Attention: {config["name"]}', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Features (To)')
            ax1.set_ylabel('Features (From)')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Extract and visualize sample attention
            predictions, attention_matrices = model.predict(
                X_test,
                return_attention=True,
                attention_layer=config["attention_layer"],
                attention_head=config["attention_head"],
                attention_aggregation=config["attention_aggregation"],
                attention_type="items"
            )
            
            attention_matrix = attention_matrices[0]
            print(f"   ✓ Sample attention shape: {attention_matrix.shape}")
            
            # Process for visualization
            if len(attention_matrix.shape) == 4:
                item_attention = attention_matrix.mean(dim=0).squeeze(-1)
            else:
                item_attention = attention_matrix.squeeze()
                
            item_attention_np = item_attention.detach().cpu().numpy()
            
            # Sample attention heatmap
            train_labels = [f"Train_{j}" for j in range(len(X_train))]
            test_labels = [f"Test_{j}" for j in range(len(X_test))]
            
            sns.heatmap(
                item_attention_np,
                cmap='viridis',
                ax=ax2,
                xticklabels=train_labels[::5],  # Show every 5th label to avoid crowding
                yticklabels=test_labels
            )
            ax2.set_title(f'Sample Attention: {config["name"]}', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Training Samples')
            ax2.set_ylabel('Test Samples')
            plt.setp(ax2.get_xticklabels(), rotation=90, ha='right')
            
            plt.tight_layout()
            
            # Save figure
            safe_name = config["name"].replace(" ", "_").replace(",", "").lower()
            save_path = f"demo_{safe_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved visualization: {save_path}")
            plt.close()
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print("\nGenerated visualizations:")
    for config in demo_configs:
        safe_name = config["name"].replace(" ", "_").replace(",", "").lower()
        print(f"- demo_{safe_name}.png")
    
    print("\nKey Features Demonstrated:")
    print("✓ Layer-specific attention extraction")
    print("✓ Head-specific attention extraction") 
    print("✓ Attention aggregation methods (mean/max)")
    print("✓ Feature-wise attention with original feature names")
    print("✓ Sample-wise attention with train/test indicators")
    print("✓ Comprehensive error handling")


def custom_attention_demo():
    """Allow users to specify custom parameters."""
    print("\n" + "="*80)
    print("CUSTOM ATTENTION EXTRACTION")
    print("="*80)
    
    print("\nYou can now extract attention with custom parameters!")
    print("Example usage:")
    print("""
from enhanced_attention_visualization import TabPFNAttentionVisualizer

# After training your model:
visualizer = TabPFNAttentionVisualizer(model, feature_names, X_train, X_test, y_train, y_test)

# Extract specific layer and head:
visualizer.create_comprehensive_visualization(
    attention_layer=5,      # Specific layer (0-11, or None for all)
    attention_head=2,       # Specific head (0-N, or None for all)
    attention_aggregation="max",  # "mean" or "max"
    save_prefix="my_analysis"
)

# Compare multiple layers:
visualizer.compare_layers(
    layers=[0, 5, 11],
    attention_type="features",  # "features" or "items"
    save_path="layer_comparison.png"
)

# Compare multiple heads:
visualizer.compare_heads(
    heads=[0, 1, 2],
    attention_layer=5,
    save_path="head_comparison.png"
)
""")


if __name__ == "__main__":
    demo_attention_extraction()
    custom_attention_demo()