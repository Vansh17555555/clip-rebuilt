import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    """
    CLIPLoss implements the symmetric InfoNCE (Information Noise-Contrastive Estimation) loss.
    It measures the alignment between matched text-image pairs (positive pairs) while
    minimizing alignment for unmatched pairings (negative pairs) within a batch.
    
    Mathematical Formulation:
        Let B be the batch size.
        Let I be the L2-normalized image embeddings (shape: [B, D]).
        Let T be the L2-normalized text embeddings (shape: [B, D]).
        Let t be the temperature scale parameter.
        
        1. Pairwise Similarity Matrix (logits):
           S = t * (I . T^T)   (shape: [B, B])
           where S[i, j] is the scaled cosine similarity between image i and text j.
           
        2. Targets:
           Since the image at index i matches the text at index i, the ground truth targets
           are a simple sequence: [0, 1, 2, ..., B-1].
           
        3. Image-to-Text Loss (Row-wise Cross-Entropy):
           loss_i2t = CE(S, targets)
           
        4. Text-to-Image Loss (Column-wise Cross-Entropy):
           loss_t2i = CE(S^T, targets)
           
        5. Symmetric Loss:
           Loss = (loss_i2t + loss_t2i) / 2
    """
    def __init__(self):
        super().__init__()

    def forward(self, 
                image_embeddings: torch.Tensor, 
                text_embeddings: torch.Tensor, 
                temperature: torch.Tensor) -> torch.Tensor:
        """
        Computes the contrastive loss.
        
        Args:
            image_embeddings: Normalized image representations [Batch_Size, shared_dim]
            text_embeddings: Normalized text representations [Batch_Size, shared_dim]
            temperature: Scalar tensor containing the scaling factor
            
        Returns:
            Symmetric contrastive loss (scalar tensor)
        """
        batch_size = image_embeddings.size(0)
        device = image_embeddings.device
        
        # 1. Compute cosine similarity matrix -> [Batch_Size, Batch_Size]
        # Since embeddings are already L2 normalized, matrix multiplication 
        # (I . T^T) is mathematically identical to pairwise cosine similarities:
        # CosSim(x, y) = (x . y) / (||x|| * ||y||) = x . y (when ||x|| = ||y|| = 1)
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.T)
        
        # 2. Scale similarity matrix by temperature
        logits = similarity_matrix * temperature
        
        # 3. Define target indices -> [0, 1, 2, ..., Batch_Size - 1]
        # Diagonal items are the matching (positive) pairs
        targets = torch.arange(batch_size, device=device)
        
        # 4. Calculate Image-to-Text Classification Loss (horizontal search)
        # Classifies which of the captions matches a given image
        loss_i2t = F.cross_entropy(logits, targets)
        
        # 5. Calculate Text-to-Image Classification Loss (vertical search)
        # Classifies which of the images matches a given caption
        loss_t2i = F.cross_entropy(logits.T, targets)
        
        # 6. Take symmetric average
        symmetric_loss = (loss_i2t + loss_t2i) / 2.0
        
        return symmetric_loss

# Quick Sanity Test
if __name__ == "__main__":
    B, D = 4, 128
    loss_fn = CLIPLoss()
    
    # Generate random L2-normalized embeddings
    img_embeds = F.normalize(torch.randn(B, D), p=2, dim=-1)
    txt_embeds = F.normalize(torch.randn(B, D), p=2, dim=-1)
    
    # Initialize temperature (e.g., 14.28 scale factor)
    temp = torch.tensor(14.285)
    
    loss = loss_fn(img_embeds, txt_embeds, temp)
    print(f"Computed Loss: {loss.item():.4f}")
    
    # Analytical verification:
    # If embeddings were perfectly aligned (identity matrix):
    perfect_img = torch.eye(B)
    perfect_txt = torch.eye(B)
    # Cosine similarities will be 1 on diagonal, 0 on off-diagonal
    perfect_loss = loss_fn(perfect_img, perfect_txt, torch.tensor(1.0))
    print(f"Perfect similarity loss (temp=1): {perfect_loss.item():.4f}")
