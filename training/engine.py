import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Any, Union
from clip_from_scratch.training.metrics import calculate_recalls

def train_one_epoch(model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    loss_fn: nn.Module, 
                    scheduler: Any, 
                    device: torch.device, 
                    epoch: int, 
                    log_interval: int = 20) -> float:
    """
    Trains the CLIP model for a single epoch.
    
    Args:
        model: The CLIP model.
        dataloader: Training DataLoader.
        optimizer: Optimizer.
        loss_fn: CLIPLoss module.
        scheduler: Learning rate scheduler.
        device: CUDA or CPU device.
        epoch: Current epoch index (for logging).
        log_interval: Step frequency for intermediate logging.
        
    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)
    
    progress_bar = tqdm(enumerate(dataloader), total=total_batches, desc=f"Epoch {epoch} [Train]")
    
    for step, batch in progress_bar:
        # Move inputs to device
        images = batch["image"].to(device)
        captions = batch["caption"].to(device)
        
        # Zero gradient buffers
        optimizer.zero_grad()
        
        # Forward pass through CLIP
        outputs = model(images, captions)
        img_embeds = outputs["image_embeddings"]
        txt_embeds = outputs["text_embeddings"]
        temperature = outputs["temperature"]
        
        # Compute symmetric InfoNCE loss
        loss = loss_fn(img_embeds, txt_embeds, temperature)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients (critical for recurrent networks like LSTMs)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Accumulate metrics
        running_loss += loss.item()
        
        # Update progress bar
        if step % log_interval == 0 or step == total_batches - 1:
            avg_step_loss = running_loss / (step + 1)
            progress_bar.set_postfix({
                "loss": f"{avg_step_loss:.4f}",
                "temp": f"{temperature.item():.2f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            
    # Step the learning rate scheduler if it's epoch-based
    if scheduler is not None and not isinstance(scheduler, str):
        # We step after the epoch, or let train.py handle it based on scheduler type
        pass
        
    return running_loss / total_batches

@torch.no_grad()
def evaluate(model: nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             loss_fn: nn.Module, 
             device: torch.device) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Evaluates the CLIP model on validation/test data.
    Gathers all embeddings from the validation loader and computes global,
    high-quality retrieval metrics (Recall@1, @5, @10) across the whole dataset.
    
    Args:
        model: The CLIP model.
        dataloader: Validation/Test DataLoader.
        loss_fn: CLIPLoss module.
        device: CUDA or CPU device.
        
    Returns:
        Dictionary containing validation loss and cross-modal recalls.
    """
    model.eval()
    running_loss = 0.0
    total_batches = len(dataloader)
    
    all_image_embeddings = []
    all_text_embeddings = []
    
    progress_bar = tqdm(dataloader, total=total_batches, desc="Evaluating")
    
    for batch in progress_bar:
        images = batch["image"].to(device)
        captions = batch["caption"].to(device)
        
        # Forward pass
        outputs = model(images, captions)
        img_embeds = outputs["image_embeddings"]
        txt_embeds = outputs["text_embeddings"]
        temperature = outputs["temperature"]
        
        # Compute batch loss
        loss = loss_fn(img_embeds, txt_embeds, temperature)
        running_loss += loss.item()
        
        # Gather all batch embeddings for global retrieval evaluation
        all_image_embeddings.append(img_embeds.cpu())
        all_text_embeddings.append(txt_embeds.cpu())
        
    # Concatenate all gathered embeddings -> shapes: [N_samples, shared_dim]
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    # Calculate global cosine similarity matrix: [N_samples, N_samples]
    # Since embeddings are L2 normalized, CosSim = I . T^T
    global_similarity = torch.matmul(all_image_embeddings, all_text_embeddings.T)
    
    # Compute Recall metrics
    recalls = calculate_recalls(global_similarity)
    
    avg_loss = running_loss / total_batches
    
    return {
        "val_loss": avg_loss,
        "metrics": recalls
    }
