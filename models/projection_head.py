import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    ProjectionHead maps the raw high-dimensional embeddings from the individual image
    and text encoders into a shared, lower-dimensional contrastive latent space.
    
    Structure:
        Linear -> LayerNorm -> GELU -> Linear -> Dropout (Optional)
    """
    def __init__(self, in_features: int, projection_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(in_features, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Residual skip connection if dimensions align or if we project the input
        self.fc_residual = nn.Linear(in_features, projection_dim) if in_features != projection_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw encoder embedding tensor of shape [Batch_Size, in_features]
            
        Returns:
            Projected embedding tensor of shape [Batch_Size, projection_dim]
        """
        # Save residual path
        residual = self.fc_residual(x)
        
        # Forward through projection MLP
        projected = self.projection(x)
        
        # Add residual connection and apply LayerNorm + Dropout
        out = self.layer_norm(projected + residual)
        return self.dropout(out)

# Quick Sanity Test
if __name__ == "__main__":
    B, D_in, D_out = 4, 512, 256
    proj = ProjectionHead(in_features=D_in, projection_dim=D_out)
    dummy_input = torch.randn(B, D_in)
    dummy_output = proj(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")  # Should be [4, 256]
