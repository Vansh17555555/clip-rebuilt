import torch
import torch.nn as nn
import torchvision.models as models
from clip_from_scratch.models.projection_head import ProjectionHead

class ImageEncoder(nn.Module):
    """
    ImageEncoder extracts feature representations from input images using a ResNet-18 backbone,
    projects them to a shared embedding space, and normalizes them.
    
    Architecture:
        ResNet18 Backbone (No Classifier Head) -> ProjectionHead -> L2 Normalization
    """
    def __init__(self, 
                 projection_dim: int = 256, 
                 pretrained: bool = False, 
                 dropout: float = 0.1):
        super().__init__()
        
        # Load standard ResNet18 backbone
        # Note: we use weights=None or weights=ResNet18_Weights.DEFAULT depending on PyTorch version.
        # To maintain compatibility across standard PyTorch environments, we handle both.
        try:
            if pretrained:
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                resnet = models.resnet18(weights=None)
        except AttributeError:
            # Fallback for older torchvision versions
            resnet = models.resnet18(pretrained=pretrained)
            
        self.feature_dim = resnet.fc.in_features  # Typically 512 for ResNet18
        
        # Strip away the final classification fully connected layer
        # Re-use ResNet features up to the global average pooling layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Custom projection head mapping resnet features to the shared space
        self.projection_head = ProjectionHead(
            in_features=self.feature_dim, 
            projection_dim=projection_dim, 
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Preprocessed image batch tensor of shape [Batch_Size, 3, 224, 224]
            
        Returns:
            L2-normalized projected embeddings of shape [Batch_Size, projection_dim]
        """
        # Extract features through ResNet [Batch_Size, 512, 1, 1]
        features = self.backbone(x)
        
        # Flatten [Batch_Size, 512]
        features = features.view(features.size(0), -1)
        
        # Project to shared space [Batch_Size, projection_dim]
        projected = self.projection_head(features)
        
        # L2 Normalize embeddings: z = x / ||x||_2
        normalized_embeddings = nn.functional.normalize(projected, p=2, dim=-1)
        
        return normalized_embeddings

# Quick Sanity Test
if __name__ == "__main__":
    B = 2
    encoder = ImageEncoder(projection_dim=256, pretrained=False)
    dummy_images = torch.randn(B, 3, 224, 224)
    embeddings = encoder(dummy_images)
    print(f"Input image shape: {dummy_images.shape}")
    print(f"Output embedding shape: {embeddings.shape}")  # [2, 256]
    
    # Check L2 Normalization (norm should be close to 1)
    norms = torch.norm(embeddings, p=2, dim=-1)
    print(f"Embedding norms: {norms.tolist()}")  # should be [1.0, 1.0]
