#!/usr/bin/env python3
"""
Comprehensive Enhanced Attention Visualization for TabPFN

This script provides complete visualization capabilities for TabPFN attention maps with all requested configurations:

1. One layer one head: attention_layers=i, attention_head=j
2. Multi or all layers multi or all heads: attention_layers=List or None, attention_head=List or None, attention_aggregation="mean" or "max"
3. One layer multi/all heads with attention_aggregation
4. Multi/all layers one head with attention_aggregation

Supports both attention_type="features" and attention_type="items"
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Optional, Literal, List, Tuple, Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')

from src.tabpfn.regressor import TabPFNRegressor


class ComprehensiveTabPFNAttentionVisualizer:
    """Comprehensive attention visualizer for TabPFN models with all requested configurations."""
    
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
        
        # Create sample labels
        self.train_sample_labels = [f"Train_{i}" for i in range(len(X_train))]
        self.test_sample_labels = [f"Test_{i}" for i in range(len(X_test))]
        
    def extract_attention(self, 
                         attention_layers: Optional[Union[int, List[int]]] = None,
                         attention_head: Optional[Union[int, List[int]]] = None,
                         attention_aggregation: Literal["mean", "max"] = "mean",
                         attention_type: Literal["features", "items"] = "features") -> torch.Tensor:
        """
        Extract attention matrices from the model with flexible layer/head specification.
        
        Args:
            attention_layers: Specific layer(s) to extract attention from (None for all layers)
            attention_head: Specific attention head(s) to extract (None for all heads)
            attention_aggregation: Method to aggregate attention across heads/layers
            attention_type: Type of attention to extract ("features" or "items")
            
        Returns:
            Attention tensor
        """
        # Handle single layer specification
        if isinstance(attention_layers, int):
            attention_layer = attention_layers
        elif isinstance(attention_layers, list) and len(attention_layers) == 1:
            attention_layer = attention_layers[0]
        else:
            attention_layer = None
            
        # Handle single head specification  
        if isinstance(attention_head, int):
            head = attention_head
        elif isinstance(attention_head, list) and len(attention_head) == 1:
            head = attention_head[0]
        else:
            head = None
            
        predictions, attention_matrices = self.model.predict(
            self.X_test,
            return_attention=True,
            attention_layer=attention_layer,
            attention_head=head,
            attention_aggregation=attention_aggregation,
            attention_type=attention_type
        )
        
        # Handle multiple layers/heads aggregation if needed
        if isinstance(attention_layers, list) and len(attention_layers) > 1:
            # Extract attention from multiple specific layers
            all_layer_attentions = []
            for layer in attention_layers:
                _, layer_attention = self.model.predict(
                    self.X_test,
                    return_attention=True,
                    attention_layer=layer,
                    attention_head=head,
                    attention_aggregation=attention_aggregation,
                    attention_type=attention_type
                )
                all_layer_attentions.append(layer_attention[0])
            
            # Aggregate across layers
            stacked_attention = torch.stack(all_layer_attentions, dim=0)
            if attention_aggregation == "mean":
                attention_matrices = [stacked_attention.mean(dim=0)]
            else:  # max
                attention_matrices = [stacked_attention.max(dim=0)[0]]
        
        return attention_matrices[0]
    
    def _process_attention_matrix(self, attention_matrix: torch.Tensor, 
                                attention_type: Literal["features", "items"] = "features") -> np.ndarray:
        """
        Process the raw attention matrix to get the proper format for visualization.
        
        Args:
            attention_matrix: Raw attention matrix from model
            attention_type: Type of attention ("features" or "items")
            
        Returns:
            Processed numpy array ready for visualization
        """
        # Handle different tensor shapes
        if len(attention_matrix.shape) == 4:
            # Shape: [batch, heads, seq, seq] or similar
            processed = attention_matrix.mean(dim=0).squeeze(-1)
            if len(processed.shape) == 3:
                processed = processed.mean(dim=0)
        elif len(attention_matrix.shape) == 3:
            # Shape: [heads, seq, seq] or [seq, seq, 1]
            if attention_matrix.shape[-1] == 1:
                processed = attention_matrix.squeeze(-1)
            else:
                processed = attention_matrix.mean(dim=0)
        else:
            processed = attention_matrix.squeeze()
            
        # Convert to numpy
        processed_np = processed.detach().cpu().numpy()
        
        # For features, we need to handle the dimension mismatch
        if attention_type == "features":
            actual_size = processed_np.shape[0]
            expected_size = len(self.feature_names)
            
            if actual_size < expected_size:
                # Pad the matrix to match expected feature count
                padded_matrix = np.zeros((expected_size, expected_size))
                padded_matrix[:actual_size, :actual_size] = processed_np
                processed_np = padded_matrix
                print(f"Info: Attention matrix padded from {actual_size}x{actual_size} to {expected_size}x{expected_size}")
            elif actual_size > expected_size:
                # Truncate the matrix if it's larger than expected
                processed_np = processed_np[:expected_size, :expected_size]
                print(f"Info: Attention matrix truncated from {actual_size}x{actual_size} to {expected_size}x{expected_size}")
        
        return processed_np
    
    def visualize_attention(self,
                          attention_layers: Optional[Union[int, List[int]]] = None,
                          attention_head: Optional[Union[int, List[int]]] = None,
                          attention_aggregation: Literal["mean", "max"] = "mean",
                          attention_type: Literal["features", "items"] = "features",
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Visualize attention maps with flexible configuration.
        
        Args:
            attention_layers: Specific layer(s) to extract attention from
            attention_head: Specific attention head(s) to extract  
            attention_aggregation: Method to aggregate attention across heads/layers
            attention_type: Type of attention to extract ("features" or "items")
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        attention_matrix = self.extract_attention(
            attention_layers=attention_layers,
            attention_head=attention_head, 
            attention_aggregation=attention_aggregation,
            attention_type=attention_type
        )
        
        # Process attention matrix
        processed_attention_np = self._process_attention_matrix(attention_matrix, attention_type)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up labels and title based on attention type
        if attention_type == "features":
            labels = self.feature_names[:processed_attention_np.shape[0]]
            xlabel = 'Features (To)'
            ylabel = 'Features (From)'
            title_prefix = 'Feature-wise'
        else:  # items
            labels = None
            xlabel = 'Training Samples'
            ylabel = 'Test Samples'
            title_prefix = 'Sample-wise'
            
        # Create heatmap
        sns.heatmap(
            processed_attention_np,
            annot=attention_type == "features",  # Only annotate for features
            fmt='.3f' if attention_type == "features" else '.2f',
            cmap='viridis',
            ax=ax,
            cbar=True,
            xticklabels=labels if attention_type == "features" else self.train_sample_labels[:processed_attention_np.shape[1]],
            yticklabels=labels if attention_type == "features" else self.test_sample_labels[:processed_attention_np.shape[0]]
        )
        
        # Create title
        layer_str = self._format_layer_string(attention_layers)
        head_str = self._format_head_string(attention_head, attention_aggregation)
        ax.set_title(f'{title_prefix} Attention Map\n{layer_str}, {head_str}', fontsize=14, fontweight='bold')
        
        # Set labels
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # Rotate labels for better readability
        if attention_type == "features":
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        else:
            plt.xticks(rotation=90, ha='right')
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved {attention_type} attention visualization: {save_path}")
            
        return fig
    
    def _format_layer_string(self, attention_layers: Optional[Union[int, List[int]]]) -> str:
        """Format layer specification for titles."""
        if attention_layers is None:
            return "All Layers"
        elif isinstance(attention_layers, int):
            return f"Layer {attention_layers}"
        elif isinstance(attention_layers, list):
            if len(attention_layers) == 1:
                return f"Layer {attention_layers[0]}"
            else:
                return f"Layers {', '.join(map(str, attention_layers))}"
        return "Unknown Layers"
    
    def _format_head_string(self, attention_head: Optional[Union[int, List[int]]], 
                           attention_aggregation: str) -> str:
        """Format head specification for titles."""
        if attention_head is None:
            return f"All Heads ({attention_aggregation})"
        elif isinstance(attention_head, int):
            return f"Head {attention_head}"
        elif isinstance(attention_head, list):
            if len(attention_head) == 1:
                return f"Head {attention_head[0]}"
            else:
                return f"Heads {', '.join(map(str, attention_head))} ({attention_aggregation})"
        return "Unknown Heads"
    
    def create_comprehensive_visualization(self,
                                         attention_layers: Optional[Union[int, List[int]]] = None,
                                         attention_head: Optional[Union[int, List[int]]] = None,
                                         attention_aggregation: Literal["mean", "max"] = "mean",
                                         save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create comprehensive attention visualizations for both features and items.
        
        Args:
            attention_layers: Specific layer(s) to extract attention from
            attention_head: Specific attention head(s) to extract
            attention_aggregation: Method to aggregate attention across heads/layers
            save_path: Base path for saved files (will append _features.png and _items.png)
            
        Returns:
            Dictionary of figure objects
        """
        figures = {}
        
        # Feature attention
        print("Creating feature attention visualization...")
        fig_features = self.visualize_attention(
            attention_layers=attention_layers,
            attention_head=attention_head,
            attention_aggregation=attention_aggregation,
            attention_type="features",
            save_path=f"{save_path}_features.png" if save_path else None
        )
        figures['features'] = fig_features
        
        # Sample attention  
        print("Creating sample attention visualization...")
        fig_samples = self.visualize_attention(
            attention_layers=attention_layers,
            attention_head=attention_head,
            attention_aggregation=attention_aggregation,
            attention_type="items",
            save_path=f"{save_path}_items.png" if save_path else None
        )
        figures['items'] = fig_samples
        
        return figures


def demonstrate_all_configurations():
    """Demonstrate all requested attention visualization configurations."""
    print("="*80)
    print("Comprehensive Enhanced TabPFN Attention Visualization")
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
    print("\n3. Creating comprehensive attention visualizer...")
    visualizer = ComprehensiveTabPFNAttentionVisualizer(
        model=model,
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
    
    # Configuration 1: One layer one head
    print("\n4. Configuration 1: One layer one head")
    print("4.1. Layer 5, Head 0")
    visualizer.create_comprehensive_visualization(
        attention_layers=5,
        attention_head=0,
        save_path="config1_layer5_head0"
    )
    
    print("4.2. Layer 11, Head 1")
    visualizer.create_comprehensive_visualization(
        attention_layers=11,
        attention_head=1,
        save_path="config1_layer11_head1"
    )
    
    # Configuration 2: Multi/all layers multi/all heads with aggregation
    print("\n5. Configuration 2: Multi/all layers multi/all heads with aggregation")
    print("5.1. All layers, all heads (mean aggregation)")
    visualizer.create_comprehensive_visualization(
        attention_layers=None,
        attention_head=None,
        attention_aggregation="mean",
        save_path="config2_all_layers_all_heads_mean"
    )
    
    print("5.2. All layers, all heads (max aggregation)")
    visualizer.create_comprehensive_visualization(
        attention_layers=None,
        attention_head=None,
        attention_aggregation="max",
        save_path="config2_all_layers_all_heads_max"
    )
    
    print("5.3. Multiple layers [0, 5, 11], all heads (mean aggregation)")
    visualizer.create_comprehensive_visualization(
        attention_layers=[0, 5, 11],
        attention_head=None,
        attention_aggregation="mean",
        save_path="config2_multi_layers_all_heads_mean"
    )
    
    # Configuration 3: One layer multi/all heads with aggregation
    print("\n6. Configuration 3: One layer multi/all heads with aggregation")
    print("6.1. Layer 5, all heads (mean aggregation)")
    visualizer.create_comprehensive_visualization(
        attention_layers=5,
        attention_head=None,
        attention_aggregation="mean",
        save_path="config3_layer5_all_heads_mean"
    )
    
    print("6.2. Layer 11, all heads (max aggregation)")
    visualizer.create_comprehensive_visualization(
        attention_layers=11,
        attention_head=None,
        attention_aggregation="max",
        save_path="config3_layer11_all_heads_max"
    )
    
    # Configuration 4: Multi/all layers one head with aggregation
    print("\n7. Configuration 4: Multi/all layers one head with aggregation")
    print("7.1. All layers, head 0 (mean aggregation)")
    visualizer.create_comprehensive_visualization(
        attention_layers=None,
        attention_head=0,
        attention_aggregation="mean",
        save_path="config4_all_layers_head0_mean"
    )
    
    print("7.2. Multiple layers [0, 5, 11], head 1 (max aggregation)")
    visualizer.create_comprehensive_visualization(
        attention_layers=[0, 5, 11],
        attention_head=1,
        attention_aggregation="max",
        save_path="config4_multi_layers_head1_max"
    )
    
    print("\n" + "="*80)
    print("Comprehensive Enhanced Attention Visualization Complete!")
    print("="*80)
    print("\nGenerated visualizations for all configurations:")
    print("Configuration 1 (One layer one head):")
    print("- config1_layer5_head0_features.png / config1_layer5_head0_items.png")
    print("- config1_layer11_head1_features.png / config1_layer11_head1_items.png")
    print("\nConfiguration 2 (Multi/all layers multi/all heads):")
    print("- config2_all_layers_all_heads_mean_features.png / config2_all_layers_all_heads_mean_items.png")
    print("- config2_all_layers_all_heads_max_features.png / config2_all_layers_all_heads_max_items.png")
    print("- config2_multi_layers_all_heads_mean_features.png / config2_multi_layers_all_heads_mean_items.png")
    print("\nConfiguration 3 (One layer multi/all heads):")
    print("- config3_layer5_all_heads_mean_features.png / config3_layer5_all_heads_mean_items.png")
    print("- config3_layer11_all_heads_max_features.png / config3_layer11_all_heads_max_items.png")
    print("\nConfiguration 4 (Multi/all layers one head):")
    print("- config4_all_layers_head0_mean_features.png / config4_all_layers_head0_mean_items.png")
    print("- config4_multi_layers_head1_max_features.png / config4_multi_layers_head1_max_items.png")


if __name__ == "__main__":
    demonstrate_all_configurations()