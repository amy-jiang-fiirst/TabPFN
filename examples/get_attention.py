import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.datasets import fetch_california_housing

from tabpfn import TabPFNRegressor

def visualize_attention():
    # 1. Load data
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    # Using a subset for quicker demonstration
    X = X[:200]
    y = y[:200]
    feature_names = X.columns.tolist()

    # 2. Initialize and fit TabPFNRegressor
    # Using fewer estimators for speed in an example.
    # Set device to 'cpu' for reproducibility in example, change if GPU is available/preferred.
    model = TabPFNRegressor(device="cpu", n_estimators=2)
    model.fit(X, y)

    # 3. Get predictions and attention matrices
    # We need to predict on some data. Can be the training data itself or a test set.
    # Using a small subset of X for prediction to keep attention matrices manageable.
    X_pred = X[:5] # Using 5 samples for prediction
    
    # The predict method now returns a tuple: (predictions, list_of_attention_tensors)
    # when return_attention=True.
    # Each tensor in list_of_attention_tensors corresponds to an estimator.
    # The tensor itself is the attention probabilities from the *last* layer of that estimator.
    try:
        predictions, attention_list = model.predict(X_pred, return_attention=True)
    except Exception as e:
        print(f"Error during predict with return_attention=True: {e}")
        print("This might indicate that the changes to TabPFNRegressor.predict were not correctly picked up.")
        print("Or, the shape/nature of attention_list is not as expected.")
        return

    if not attention_list:
        print("Attention list is empty. No attention matrices to visualize.")
        return

    # 4. Process the attention matrix
    # We'll use the attention from the first estimator for this example.
    # The attention_list contains tensors which might be None if something went wrong
    # or if attention couldn't be extracted (e.g. flash attention not available/configured for probs)
    raw_attention_matrix = attention_list[0]

    if raw_attention_matrix is None:
        print("The first attention matrix is None. Cannot visualize.")
        print("This might happen if `ps` was not captured correctly in MultiHeadAttention for certain paths (e.g. Flash Attention).")
        return

    print(f"Raw attention matrix shape: {raw_attention_matrix.shape}")

    # The exact shape of raw_attention_matrix depends on the specifics of PerFeatureTransformer's forward pass.
    # It's expected to be (batch_size_of_X_pred, num_heads, num_tokens_q, num_tokens_kv)
    # For feature attention, num_tokens_q and num_tokens_kv should be (num_features + 1)
    # where +1 is for the target variable y.

    # We expect num_tokens_q and num_tokens_kv to be n_features + 1 (for y)
    # The tokens are [feature1, feature2, ..., featureN, target_y]
    expected_num_tokens = X.shape[1] + 1 
    
    # Let's check the last two dimensions.
    if raw_attention_matrix.shape[-1] != expected_num_tokens or \
       raw_attention_matrix.shape[-2] != expected_num_tokens:
        print(f"Warning: Unexpected attention matrix dimensions.")
        print(f"Expected last two dims: ({expected_num_tokens}, {expected_num_tokens})")
        print(f"Got: ({raw_attention_matrix.shape[-2]}, {raw_attention_matrix.shape[-1]})")
        print("The visualization might not represent feature-feature attention correctly.")
        # Attempt to select the part that might correspond to feature attention if the sequence is longer
        # This is a heuristic. The actual structure might be more complex if other types of attention
        # (e.g., attention over samples) are mixed in the final layer's output.
        # For now, we assume the last `expected_num_tokens` are the relevant ones if the matrix is larger.
        # This part needs to be robust based on understanding what `final_attention_probs` truly represents.
        # Based on `PerFeatureTransformer._forward`, `encoder_out` (and thus `final_attention_probs`)
        # comes from processing `embedded_input` of shape (b, s, f+1, e).
        # The attention layers operate on the last two dimensions effectively.
        # So, the (f+1) part is what we are interested in.
        # The `ps` from `MultiHeadAttention.compute_attention_heads` has shape (batch, nhead, seqlen_q, seqlen_kv).
        # In `PerFeatureEncoderLayer.forward`, if `attn_between_features` is active,
        # `seqlen_q` and `seqlen_kv` are `src.shape[2]` which is `num_feature_blocks` (num_features_grouped + 1 for target).
        # The batch dimension for the attention layer call in PerFeatureEncoderLayer is X_pred.shape[0].
        
        attention_matrix_for_sample = raw_attention_matrix[0] # Take attention for the first sample in X_pred
    else:
        attention_matrix_for_sample = raw_attention_matrix[0]


    # Average across heads
    if attention_matrix_for_sample.ndim == 3: # (num_heads, num_tokens, num_tokens)
        averaged_attention = attention_matrix_for_sample.mean(axis=0)
    elif attention_matrix_for_sample.ndim == 2: # (num_tokens, num_tokens) - already averaged or single head
        averaged_attention = attention_matrix_for_sample
    else:
        print(f"Attention matrix for sample has unexpected ndim: {attention_matrix_for_sample.ndim}. Shape: {attention_matrix_for_sample.shape}")
        return

    # Convert to NumPy array
    if isinstance(averaged_attention, torch.Tensor):
        averaged_attention_np = averaged_attention.cpu().numpy()
    else:
        averaged_attention_np = averaged_attention # If it's already numpy

    # Ensure it's a 2D matrix for heatmap
    if averaged_attention_np.ndim != 2 or averaged_attention_np.shape[0] != averaged_attention_np.shape[1]:
        print(f"Averaged attention is not a square 2D matrix. Shape: {averaged_attention_np.shape}")
        return
        
    if averaged_attention_np.shape[0] != expected_num_tokens:
        print(f"Warning: Processed attention matrix dimension ({averaged_attention_np.shape[0]}) does not match expected_num_tokens ({expected_num_tokens}). Cropping/padding for visualization.")
        # Simple crop / pad for visualization if mismatch
        size = expected_num_tokens
        final_matrix_for_vis = np.zeros((size, size))
        crop_size = min(size, averaged_attention_np.shape[0])
        final_matrix_for_vis[:crop_size, :crop_size] = averaged_attention_np[:crop_size, :crop_size]
    else:
        final_matrix_for_vis = averaged_attention_np


    # 5. Visualize the heatmap
    plt.figure(figsize=(10, 8))
    heatmap_labels = feature_names + ["Target_y"] # Add target to labels
    sns.heatmap(final_matrix_for_vis, annot=True, cmap="viridis", fmt=".2f",
                xticklabels=heatmap_labels, yticklabels=heatmap_labels)
    plt.title("Feature Attention Matrix Heatmap (Last Layer, Averaged Heads, First Estimator, First Prediction Sample)")
    plt.xlabel("Key Features (attended to)")
    plt.ylabel("Query Features (attending from)")
    plt.tight_layout()
    
    # Save or show the plot
    plot_filename = "feature_attention_heatmap.png"
    plt.savefig(plot_filename)
    print(f"Saved heatmap to {plot_filename}")
    # plt.show() # Uncomment to display interactively

if __name__ == "__main__":
    visualize_attention()
