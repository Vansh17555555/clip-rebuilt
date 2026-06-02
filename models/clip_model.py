import math
import torch
import torch.nn as nn
from clip_from_scratch.models.image_encoder import ImageEncoder
from clip_from_scratch.models.text_encoder import TextEncoder

class CLIPModel(nn.Module):
    """
    CLIPModel combines the ImageEncoder and TextEncoder into a unified multimodal network.
    It manages the shared embedding space projections and contains a learnable,
    log-scaled temperature parameter to calibrate the contrastive logits scaling.
    """
    def __init__(self, 
                 vocab_size: int, 
                 pad_idx: int = 0,
                 image_pretrained: bool = False,
                 feature_dim: int = 512,
                 word_embedding_dim: int = 128,
                 lstm_hidden_dim: int = 128,
                 lstm_num_layers: int = 2,
                 lstm_bidirectional: bool = True,
                 projection_dim: int = 256,
                 temperature_init: float = 0.07,
                 dropout: float = 0.1):
        super().__init__()
        
        # 1. Initialize Image Encoder (ResNet-18 Backbone)
        self.image_encoder = ImageEncoder(
            projection_dim=projection_dim,
            pretrained=image_pretrained,
            dropout=dropout
        )
        
        # 2. Initialize Text Encoder (BiLSTM Backbone)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            word_embedding_dim=word_embedding_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            lstm_bidirectional=lstm_bidirectional,
            projection_dim=projection_dim,
            pad_idx=pad_idx,
            dropout=dropout
        )
        
        # 3. Trainable Temperature Scale: Log-space implementation avoids negativity and division instability.
        # Initialized to ln(1 / temperature_init), so exp(log_temperature) matches (1 / temperature_init).
        # E.g., for temperature_init = 0.07, log_temp = ln(14.2857) = 2.659
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature_init), dtype=torch.float32)
        )

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> dict:
        """
        Runs the forward pass for a batch of images and corresponding token sequences.
        
        Args:
            images: Tensor of shape [Batch_Size, 3, 224, 224]
            texts: Tensor of shape [Batch_Size, Sequence_Length]
            
        Returns:
            A dictionary containing:
                "image_embeddings": L2-normalized image embeddings [Batch_Size, projection_dim]
                "text_embeddings": L2-normalized text embeddings [Batch_Size, projection_dim]
                "temperature": Clamped temperature scaling factor (scalar)
        """
        # Encode and normalize images
        image_embeddings = self.image_encoder(images)
        
        # Encode and normalize texts
        text_embeddings = self.text_encoder(texts)
        
        # Compute scaling factor: e^(log_temperature)
        # Standard CLIP clamps temperature scale (exp(log_temp)) to a max value of 100
        # to prevent numerical instability in soft-max (entropy collapse)
        clamped_log_temp = torch.clamp(self.log_temperature, max=math.log(100.0))
        temperature = clamped_log_temp.exp()
        
        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "temperature": temperature
        }

# Quick Sanity Test
if __name__ == "__main__":
    B, S, V = 2, 10, 100
    model = CLIPModel(vocab_size=V, pad_idx=0, projection_dim=256, temperature_init=0.07)
    
    dummy_images = torch.randn(B, 3, 224, 224)
    dummy_tokens = torch.randint(1, V, (B, S))
    dummy_tokens[dummy_tokens == 0] = 1  # ensure no padding index for test except index 0
    
    outputs = model(dummy_images, dummy_tokens)
    print(f"Image embeddings shape: {outputs['image_embeddings'].shape}")  # [2, 256]
    print(f"Text embeddings shape: {outputs['text_embeddings'].shape}")    # [2, 256]
    print(f"Learnable temperature value: {outputs['temperature'].item():.4f}") # around 14.285
