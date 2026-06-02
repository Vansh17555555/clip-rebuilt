import torch
import torch.nn as nn
from clip_from_scratch.models.projection_head import ProjectionHead

class TextEncoder(nn.Module):
    """
    TextEncoder embeds caption token index sequences, processes them with a Bidirectional LSTM,
    pools the sequence into a sentence-level feature vector (masking padding tokens),
    projects it into the shared embedding space, and normalizes it.
    
    Architecture:
        Embedding Layer -> BiLSTM -> Masked Mean Pooling -> Projection Head -> L2 Normalization
    """
    def __init__(self, 
                 vocab_size: int, 
                 word_embedding_dim: int = 128, 
                 lstm_hidden_dim: int = 128, 
                 lstm_num_layers: int = 2, 
                 lstm_bidirectional: bool = True, 
                 projection_dim: int = 256, 
                 pad_idx: int = 0, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.pad_idx = pad_idx
        
        # Word Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=word_embedding_dim, 
            padding_idx=pad_idx
        )
        
        # Recurrent Encoder: Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=word_embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=dropout if lstm_num_layers > 1 else 0.0
        )
        
        # Determine features shape after bidirectional outputs concatenation
        self.lstm_out_dim = lstm_hidden_dim * 2 if lstm_bidirectional else lstm_hidden_dim
        
        # Custom projection head to shared space
        self.projection_head = ProjectionHead(
            in_features=self.lstm_out_dim, 
            projection_dim=projection_dim, 
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token indices batch of shape [Batch_Size, Sequence_Length]
            
        Returns:
            L2-normalized projected embeddings of shape [Batch_Size, projection_dim]
        """
        # 1. Embed token index sequences -> [Batch_Size, Sequence_Length, word_embedding_dim]
        embedded = self.embedding(x)
        
        # 2. Extract context via LSTM -> outputs shape: [Batch_Size, Sequence_Length, lstm_out_dim]
        # (h_n, c_n) are the final hidden/cell states, which we don't need since we perform pooling
        outputs, _ = self.lstm(embedded)
        
        # 3. Masked Mean Pooling: Compute mean vector over actual text tokens, ignoring pad tokens
        # Create mask: True (1) for real tokens, False (0) for pad tokens. shape: [Batch_Size, Sequence_Length]
        mask = (x != self.pad_idx).float()
        
        # Expand mask dimensions to match LSTM outputs -> [Batch_Size, Sequence_Length, 1]
        expanded_mask = mask.unsqueeze(-1)
        
        # Zero out padding token hidden states -> [Batch_Size, Sequence_Length, lstm_out_dim]
        masked_outputs = outputs * expanded_mask
        
        # Sum outputs along sequence length -> [Batch_Size, lstm_out_dim]
        summed_outputs = torch.sum(masked_outputs, dim=1)
        
        # Count actual tokens per sequence -> [Batch_Size, 1]
        # Clamp to minimum 1 to avoid division by zero on empty sequences
        token_counts = torch.clamp(torch.sum(mask, dim=1, keepdim=True), min=1.0)
        
        # Compute sentence average representation -> [Batch_Size, lstm_out_dim]
        pooled_features = summed_outputs / token_counts
        
        # 4. Project to shared space -> [Batch_Size, projection_dim]
        projected = self.projection_head(pooled_features)
        
        # 5. L2 Normalize -> [Batch_Size, projection_dim]
        normalized_embeddings = nn.functional.normalize(projected, p=2, dim=-1)
        
        return normalized_embeddings

# Quick Sanity Test
if __name__ == "__main__":
    B, S, V = 2, 10, 100
    encoder = TextEncoder(vocab_size=V, pad_idx=0, projection_dim=256)
    
    # Batch contains some padding (index 0)
    dummy_tokens = torch.tensor([
        [2, 5, 8, 12, 3, 0, 0, 0, 0, 0],
        [2, 9, 14, 15, 6, 7, 8, 10, 3, 0]
    ], dtype=torch.long)
    
    embeddings = encoder(dummy_tokens)
    print(f"Tokens input shape: {dummy_tokens.shape}")
    print(f"Output embedding shape: {embeddings.shape}")  # [2, 256]
    
    # Check L2 Normalization (norm should be close to 1)
    norms = torch.norm(embeddings, p=2, dim=-1)
    print(f"Embedding norms: {norms.tolist()}")  # should be [1.0, 1.0]
