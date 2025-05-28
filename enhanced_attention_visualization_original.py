#!/usr/bin/env python3
"""
Enhanced attention visualization for TabPFN with comprehensive features.

This script provides complete visualization capabilities for TabPFN attention maps:
- Feature-wise attention maps with original feature names and target variables
- Sample-wise attention maps with sample indices and train/test indicators
- Layer and head-specific attention extraction
- Attention head aggregation (mean/max)
- Interactive parameter selection
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Optional, Literal, List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from src.tabpfn.regressor import TabPFNRegressor


class TabPFNAttentionVisualizer:
    """Enhanced attention visualizer for TabPFN models."""
    
    def __init__(self, model: TabPFNRegressor, feature_names: List[str], 
                 X_train: np.ndarray, X_test: np.ndarray, 
                 y_train: np.ndarray, y_test: np.ndarray):
        """
        Initialize the attention visualizer.
        
        Args:
            model: Fitted TabPFN model
            feature_names: List of original feature names
            X_train: Training features
            X_test: Test features  
            y_train: Training targets
            y_test: Test targets
        """
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Create extended feature names including target
        self.extended_feature_names = feature_names + ['Target']
        
        # Create sample labels
        self.train_sample_labels = [f"Train_{i}" for i in range(len(X_train))]
        self.test_sample_labels = [f"Test_{i}" for i in range(len(X_test))]
        self.all_sample_labels = self.train_sample_labels + self.test_sample_labels
        
    def extract_attention(self, 
                         attention_layer: Optional[int] = None,
                         attention_head: Optional[int] = None,
                         attention_aggregation: Literal["mean", "max"] = "mean",
                         attention_type: Literal["features", "items"] = "features") -> torch.Tensor:
        """
        Extract attention matrices from the model.
        
        Args:
            attention_layer: Specific layer to extract attention from (None for all layers)
            attention_head: Specific attention head to extract (None for all heads)
            attention_aggregation: Method to aggregate attention across heads
            attention_type: Type of attention to extract ("features" or "items")
            
        Returns:
            Attention tensor
        """
        predictions, attention_matrices = self.model.predict(
            self.X_test,
            return_attention=True,
            attention_layer=attention_layer,
            attention_head=attention_head,
            attention_aggregation=attention_aggregation,
            attention_type=attention_type
        )
        
        return attention_matrices[0]
    
    def visualize_feature_attention(self, 
                                  attention_layer: Optional[int] = None,
                                  attention_head: Optional[int] = None,
                                  attention_aggregation: Literal["mean", "max"] = "mean",
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Visualize feature-wise attention maps.
        
        Args:
            attention_layer: Specific layer to extract attention from
            attention_head: Specific attention head to extract  
            attention_aggregation: Method to aggregate attention across heads
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        attention_matrix = self.extract_attention(
            attention_layer=attention_layer,
            attention_head=attention_head, 
            attention_aggregation=attention_aggregation,
            attention_type="features"
        )
        
        # Process attention matrix for features
        # Expected shape: [batch*seq, heads, features, 1] -> [features, features]
        if len(attention_matrix.shape) == 4:
            # Average over batch dimension and squeeze last dimension
            feature_attention = attention_matrix.mean(dim=0).squeeze(-1)
            if len(feature_attention.shape) == 3:
                # If still 3D, average over heads
                feature_attention = feature_attention.mean(dim=0)
        else:
            feature_attention = attention_matrix.squeeze()
            
        # Convert to numpy
        feature_attention_np = feature_attention.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            feature_attention_np,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            ax=ax,
            cbar=True,
            xticklabels=self.extended_feature_names,
            yticklabels=self.extended_feature_names
        )
        
        # Set title
        layer_str = f"Layer {attention_layer}" if attention_layer is not None else "All Layers"
        head_str = f"Head {attention_head}" if attention_head is not None else f"All Heads ({attention_aggregation})"
        ax.set_title(f'Feature-wise Attention Map\n{layer_str}, {head_str}', fontsize=14, fontweight='bold')
        
        # Set labels
        ax.set_xlabel('Features (To)', fontsize=12)
        ax.set_ylabel('Features (From)', fontsize=12)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved feature attention visualization: {save_path}")
            
        return fig
    
    def visualize_sample_attention(self,
                                 attention_layer: Optional[int] = None,
                                 attention_head: Optional[int] = None, 
                                 attention_aggregation: Literal["mean", "max"] = "mean",
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
        """
        Visualize sample-wise attention maps.
        
        Args:
            attention_layer: Specific layer to extract attention from
            attention_head: Specific attention head to extract
            attention_aggregation: Method to aggregate attention across heads
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        attention_matrix = self.extract_attention(
            attention_layer=attention_layer,
            attention_head=attention_head,
            attention_aggregation=attention_aggregation, 
            attention_type="items"
        )
        
        # Process attention matrix for items
        # Expected shape: [heads, test_items, train_items, 1] -> [test_items, train_items]
        if len(attention_matrix.shape) == 4:
            item_attention = attention_matrix.mean(dim=0).squeeze(-1)
        else:
            item_attention = attention_matrix.squeeze()
            
        # Convert to numpy
        item_attention_np = item_attention.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            item_attention_np,
            cmap='viridis',
            ax=ax,
            cbar=True,
            xticklabels=self.train_sample_labels,
            yticklabels=self.test_sample_labels
        )
        
        # Set title
        layer_str = f"Layer {attention_layer}" if attention_layer is not None else "All Layers"
        head_str = f"Head {attention_head}" if attention_head is not None else f"All Heads ({attention_aggregation})"
        ax.set_title(f'Sample-wise Attention Map\n{layer_str}, {head_str}', fontsize=14, fontweight='bold')
        
        # Set labels
        ax.set_xlabel('Training Samples', fontsize=12)
        ax.set_ylabel('Test Samples', fontsize=12)
        
        # Rotate labels for better readability
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved sample attention visualization: {save_path}")
            
        return fig
    
    def create_comprehensive_visualization(self,
                                         attention_layer: Optional[int] = None,
                                         attention_head: Optional[int] = None,
                                         attention_aggregation: Literal["mean", "max"] = "mean",
                                         save_prefix: str = "attention_analysis") -> Dict[str, plt.Figure]:
        """
        Create comprehensive attention visualizations.
        
        Args:
            attention_layer: Specific layer to extract attention from
            attention_head: Specific attention head to extract
            attention_aggregation: Method to aggregate attention across heads
            save_prefix: Prefix for saved files
            
        Returns:
            Dictionary of figure objects
        """
        figures = {}
        
        # Feature attention
        print("Creating feature attention visualization...")
        fig_features = self.visualize_feature_attention(
            attention_layer=attention_layer,
            attention_head=attention_head,
            attention_aggregation=attention_aggregation,
            save_path=f"{save_prefix}_features.png"
        )
        figures['features'] = fig_features
        
        # Sample attention  
        print("Creating sample attention visualization...")
        fig_samples = self.visualize_sample_attention(
            attention_layer=attention_layer,
            attention_head=attention_head,
            attention_aggregation=attention_aggregation,
            save_path=f"{save_prefix}_samples.png"
        )
        figures['samples'] = fig_samples
        
        return figures
    
    def compare_layers(self, 
                      layers: List[int],
                      attention_head: Optional[int] = None,
                      attention_aggregation: Literal["mean", "max"] = "mean",
                      attention_type: Literal["features", "items"] = "features",
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (20, 12)) -> plt.Figure:
        """
        Compare attention patterns across different layers.
        
        Args:
            layers: List of layer indices to compare
            attention_head: Specific attention head to extract
            attention_aggregation: Method to aggregate attention across heads
            attention_type: Type of attention to extract
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_layers = len(layers)
        fig, axes = plt.subplots(1, n_layers, figsize=figsize)
        if n_layers == 1:
            axes = [axes]
            
        for i, layer in enumerate(layers):
            attention_matrix = self.extract_attention(
                attention_layer=layer,
                attention_head=attention_head,
                attention_aggregation=attention_aggregation,
                attention_type=attention_type
            )
            
            # Process attention matrix
            if attention_type == "features":
                if len(attention_matrix.shape) == 4:
                    processed_attention = attention_matrix.mean(dim=0).squeeze(-1)
                    if len(processed_attention.shape) == 3:
                        processed_attention = processed_attention.mean(dim=0)
                else:
                    processed_attention = attention_matrix.squeeze()
                labels = self.extended_feature_names
            else:  # items
                if len(attention_matrix.shape) == 4:
                    processed_attention = attention_matrix.mean(dim=0).squeeze(-1)
                else:
                    processed_attention = attention_matrix.squeeze()
                labels = None
                
            processed_attention_np = processed_attention.detach().cpu().numpy()
            
            # Create heatmap
            sns.heatmap(
                processed_attention_np,
                ax=axes[i],
                cmap='viridis',
                cbar=True,
                xticklabels=labels if attention_type == "features" else False,
                yticklabels=labels if attention_type == "features" else False
            )
            
            head_str = f"Head {attention_head}" if attention_head is not None else f"All Heads ({attention_aggregation})"
            axes[i].set_title(f'Layer {layer}\n{head_str}', fontsize=12)
            
            if attention_type == "features" and labels:
                axes[i].tick_params(axis='x', rotation=45)
                
        fig.suptitle(f'{attention_type.capitalize()}-wise Attention Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved layer comparison visualization: {save_path}")
            
        return fig
    
    def compare_heads(self,
                     heads: List[int],
                     attention_layer: Optional[int] = None,
                     attention_type: Literal["features", "items"] = "features", 
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (20, 12)) -> plt.Figure:
        """
        Compare attention patterns across different heads.
        
        Args:
            heads: List of head indices to compare
            attention_layer: Specific layer to extract attention from
            attention_type: Type of attention to extract
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_heads = len(heads)
        fig, axes = plt.subplots(1, n_heads, figsize=figsize)
        if n_heads == 1:
            axes = [axes]
            
        for i, head in enumerate(heads):
            attention_matrix = self.extract_attention(
                attention_layer=attention_layer,
                attention_head=head,
                attention_aggregation="mean",  # Not used when specific head is selected
                attention_type=attention_type
            )
            
            # Process attention matrix
            if attention_type == "features":
                if len(attention_matrix.shape) == 4:
                    processed_attention = attention_matrix.mean(dim=0).squeeze(-1)
                    if len(processed_attention.shape) == 3:
                        processed_attention = processed_attention.mean(dim=0)
                else:
                    processed_attention = attention_matrix.squeeze()
                labels = self.extended_feature_names
            else:  # items
                if len(attention_matrix.shape) == 4:
                    processed_attention = attention_matrix.mean(dim=0).squeeze(-1)
                else:
                    processed_attention = attention_matrix.squeeze()
                labels = None
                
            processed_attention_np = processed_attention.detach().cpu().numpy()
            
            # Create heatmap
            sns.heatmap(
                processed_attention_np,
                ax=axes[i],
                cmap='viridis',
                cbar=True,
                xticklabels=labels if attention_type == "features" else False,
                yticklabels=labels if attention_type == "features" else False
            )
            
            layer_str = f"Layer {attention_layer}" if attention_layer is not None else "All Layers"
            axes[i].set_title(f'Head {head}\n{layer_str}', fontsize=12)
            
            if attention_type == "features" and labels:
                axes[i].tick_params(axis='x', rotation=45)
                
        fig.suptitle(f'{attention_type.capitalize()}-wise Attention Head Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved head comparison visualization: {save_path}")
            
        return fig


def main():
    """Main function demonstrating the enhanced attention visualization."""
    print("="*80)
    print("Enhanced TabPFN Attention Visualization")
    print("="*80)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    
    # Split data
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
    
    # Initialize and fit model
    print("\n2. Training TabPFN model...")
    model = TabPFNRegressor(
        n_estimators=1,
        device="cpu",
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Create visualizer
    print("\n3. Creating attention visualizer...")
    visualizer = TabPFNAttentionVisualizer(
        model=model,
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
    
    # Demonstrate different visualization capabilities
    print("\n4. Creating comprehensive visualizations...")
    
    # Example 1: All layers, all heads with mean aggregation
    print("\n4.1. All layers, all heads (mean aggregation)")
    visualizer.create_comprehensive_visualization(
        attention_layer=None,
        attention_head=None,
        attention_aggregation="mean",
        save_prefix="all_layers_all_heads_mean"
    )
    
    # Example 2: Specific layer (layer 5), all heads with max aggregation
    print("\n4.2. Layer 5, all heads (max aggregation)")
    visualizer.create_comprehensive_visualization(
        attention_layer=5,
        attention_head=None,
        attention_aggregation="max",
        save_prefix="layer5_all_heads_max"
    )
    
    # Example 3: Specific layer and head
    print("\n4.3. Layer 11, head 0")
    visualizer.create_comprehensive_visualization(
        attention_layer=11,
        attention_head=0,
        attention_aggregation="mean",  # Not used when specific head is selected
        save_prefix="layer11_head0"
    )
    
    # Example 4: Layer comparison
    print("\n4.4. Comparing layers 0, 5, and 11")
    visualizer.compare_layers(
        layers=[0, 5, 11],
        attention_head=None,
        attention_aggregation="mean",
        attention_type="features",
        save_path="layer_comparison_features.png"
    )
    
    visualizer.compare_layers(
        layers=[0, 5, 11],
        attention_head=None,
        attention_aggregation="mean",
        attention_type="items",
        save_path="layer_comparison_items.png"
    )
    
    # Example 5: Head comparison (if multiple heads exist)
    print("\n4.5. Comparing different heads in layer 5")
    try:
        visualizer.compare_heads(
            heads=[0, 1, 2],
            attention_layer=5,
            attention_type="features",
            save_path="head_comparison_features.png"
        )
    except Exception as e:
        print(f"Head comparison skipped: {e}")
    
    print("\n" + "="*80)
    print("Enhanced Attention Visualization Complete!")
    print("="*80)
    print("\nGenerated visualizations:")
    print("- all_layers_all_heads_mean_features.png")
    print("- all_layers_all_heads_mean_samples.png")
    print("- layer5_all_heads_max_features.png")
    print("- layer5_all_heads_max_samples.png")
    print("- layer11_head0_features.png")
    print("- layer11_head0_samples.png")
    print("- layer_comparison_features.png")
    print("- layer_comparison_items.png")
    print("- head_comparison_features.png (if applicable)")


if __name__ == "__main__":
    main()