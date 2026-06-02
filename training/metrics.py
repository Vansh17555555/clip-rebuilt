import torch
from typing import Dict

def calculate_recalls(similarity_matrix: torch.Tensor) -> Dict[str, float]:
    """
    Computes cross-modal retrieval metrics (Recall@1, Recall@5, Recall@10)
    for both directions: Image-to-Text (I2T) and Text-to-Image (T2I).
    
    Args:
        similarity_matrix: Cosine similarity matrix of shape [N, N]
            where row i is the cosine similarities between image i and all N captions,
            and column j is the cosine similarities between caption j and all N images.
            
    Returns:
        Dictionary containing Recall values:
        - i2t_r1, i2t_r5, i2t_r10
        - t2i_r1, t2i_r5, t2i_r10
    """
    n = similarity_matrix.size(0)
    
    # Ground truth targets: for index i, the correct target index is i
    targets = torch.arange(n, device=similarity_matrix.device).unsqueeze(1) # shape: [N, 1]
    
    # --- Image-to-Text (I2T) Retrieval ---
    # For each image (row), find the top-k most similar captions (columns)
    # Sort each row in descending order
    _, i2t_indices = torch.topk(similarity_matrix, k=min(10, n), dim=-1, largest=True, sorted=True)
    
    # Check if target caption index is in the top-k retrieved indices
    i2t_r1 = (i2t_indices[:, :1] == targets).float().mean().item()
    i2t_r5 = (i2t_indices[:, :5] == targets).any(dim=-1).float().mean().item()
    i2t_r10 = (i2t_indices[:, :10] == targets).any(dim=-1).float().mean().item()
    
    # --- Text-to-Image (T2I) Retrieval ---
    # For each caption (column), find the top-k most similar images (rows)
    # Transposing similarity matrix lets us perform row-wise top-k search as well
    similarity_matrix_t = similarity_matrix.T
    _, t2i_indices = torch.topk(similarity_matrix_t, k=min(10, n), dim=-1, largest=True, sorted=True)
    
    # Check if target image index is in the top-k retrieved indices
    t2i_r1 = (t2i_indices[:, :1] == targets).float().mean().item()
    t2i_r5 = (t2i_indices[:, :5] == targets).any(dim=-1).float().mean().item()
    t2i_r10 = (t2i_indices[:, :10] == targets).any(dim=-1).float().mean().item()
    
    return {
        "i2t_r1": i2t_r1 * 100.0,
        "i2t_r5": i2t_r5 * 100.0,
        "i2t_r10": i2t_r10 * 100.0,
        "t2i_r1": t2i_r1 * 100.0,
        "t2i_r5": t2i_r5 * 100.0,
        "t2i_r10": t2i_r10 * 100.0,
    }

# Quick Sanity Test
if __name__ == "__main__":
    # N=5 items
    # Perfect diagonal similarity
    perfect_sim = torch.eye(5)
    perfect_metrics = calculate_recalls(perfect_sim)
    print("Perfect alignment metrics:")
    for k, v in perfect_metrics.items():
        print(f"  {k}: {v:.1f}%")
        
    # Imperfect similarity
    imperfect_sim = torch.tensor([
        [0.9, 0.1, 0.2, 0.4, 0.1],  # 0 matches 0 (R1)
        [0.8, 0.2, 0.9, 0.1, 0.1],  # 1 matches 2 (not 1) - error
        [0.1, 0.1, 0.85, 0.1, 0.4], # 2 matches 2 (R1)
        [0.3, 0.2, 0.1, 0.7, 0.8],  # 3 matches 4 (not 3) - error
        [0.1, 0.1, 0.1, 0.1, 0.95], # 4 matches 4 (R1)
    ])
    imperfect_metrics = calculate_recalls(imperfect_sim)
    print("\nImperfect alignment metrics:")
    for k, v in imperfect_metrics.items():
        print(f"  {k}: {v:.1f}%")
