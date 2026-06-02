import os
import yaml
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np

# Import custom components
from clip_from_scratch.data.vocabulary import Vocabulary
from clip_from_scratch.data.tokenizer import Tokenizer
from clip_from_scratch.data.dataset import FlickrDataset, collate_fn
from clip_from_scratch.models.clip_model import CLIPModel
from clip_from_scratch.losses.contrastive_loss import CLIPLoss
from clip_from_scratch.training.engine import train_one_epoch, evaluate

def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility across runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path: str) -> dict:
    """Loads YAML configurations safely, falling back to a default dict if missing."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"[Warning] Config file not found at {config_path}. Using default configuration parameters.")
        return {
            "dataset": {
                "name": "flickr8k_dummy",
                "image_dir": "./dummy_images",
                "caption_file": "./dummy_captions.txt",
                "batch_size": 2,
                "num_workers": 0,
                "vocab_min_freq": 1,
                "max_seq_len": 12
            },
            "model": {
                "image": {
                    "pretrained": False,
                    "projection_dim": 64,
                    "feature_dim": 512
                },
                "text": {
                    "word_embedding_dim": 64,
                    "lstm_hidden_dim": 64,
                    "lstm_num_layers": 1,
                    "lstm_bidirectional": True,
                    "projection_dim": 64
                },
                "temperature_init": 0.07
            },
            "training": {
                "epochs": 3,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "lr_scheduler": "cosine",
                "device": "cpu",
                "checkpoint_dir": "./checkpoints",
                "log_interval": 1
            }
        }

def main():
    set_seed(42)
    
    # 1. Load Configurations
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
    config = load_config(config_path)
    
    # Setup Device
    device_name = config["training"].get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Ensure checkpoint directory exists
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 2. Build or Load Vocabulary
    print("Building vocabulary...")
    vocab = Vocabulary()
    
    caption_file = config["dataset"]["caption_file"]
    vocab_min_freq = config["dataset"]["vocab_min_freq"]
    
    # Read captions to build vocabulary
    sentences = []
    tokenizer_temp = Tokenizer(vocab)
    
    if os.path.exists(caption_file):
        with open(caption_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        header_skipped = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if not header_skipped and ("image,caption" in line.lower() or "image_name" in line.lower()):
                header_skipped = True
                continue
            if ',' in line:
                _, caption = line.split(',', 1)
                tokens = tokenizer_temp.tokenize(caption)
                sentences.append(tokens)
        
        vocab.build_vocabulary(sentences, min_freq=vocab_min_freq)
    else:
        # Build vocabulary using dummy text if local dataset is not present
        dummy_dataset = FlickrDataset(
            image_dir=config["dataset"]["image_dir"],
            caption_file=caption_file,
            tokenizer=tokenizer_temp
        )
        for _, _, raw_caption, _ in dummy_dataset.data_pairs:
            tokens = tokenizer_temp.tokenize(raw_caption)
            sentences.append(tokens)
        vocab.build_vocabulary(sentences, min_freq=1)
        
    print(f"Vocabulary successfully built! Total Unique Tokens: {len(vocab)}")
    
    # Save vocabulary for future evaluation
    vocab_save_path = os.path.join(checkpoint_dir, "vocab.json")
    vocab.save(vocab_save_path)
    print(f"Vocabulary saved to {vocab_save_path}")
    
    # 3. Create Tokenizer & Datasets
    tokenizer = Tokenizer(vocab)
    
    # Load entire dataset
    full_dataset = FlickrDataset(
        image_dir=config["dataset"]["image_dir"],
        caption_file=caption_file,
        tokenizer=tokenizer,
        max_seq_len=config["dataset"]["max_seq_len"],
        transform=FlickrDataset.get_default_transform(is_train=True)
    )
    
    # Perform random split for training/validation (90% Train, 10% Val)
    dataset_size = len(full_dataset)
    val_size = max(1, int(dataset_size * 0.1))
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Use validation transforms for the validation split (no training augmentations like flips/crops)
    val_dataset.dataset.transform = FlickrDataset.get_default_transform(is_train=False)
    
    print(f"Total dataset pairs: {dataset_size} | Train size: {train_size} | Val size: {val_size}")
    
    # 4. Initialize DataLoaders
    batch_size = config["dataset"]["batch_size"]
    num_workers = config["dataset"]["num_workers"]
    
    # Make sure batch_size is not larger than train_size
    if batch_size > train_size:
        batch_size = train_size
        print(f"[Info] Adjusted batch size to {batch_size} due to small dataset size.")
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True # ensures matching similarity matrix square size in InfoNCE
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    # 5. Build Model, Loss, Optimizer, and Scheduler
    model = CLIPModel(
        vocab_size=len(vocab),
        pad_idx=vocab.word2idx[vocab.pad_token],
        image_pretrained=config["model"]["image"]["pretrained"],
        feature_dim=config["model"]["image"]["feature_dim"],
        word_embedding_dim=config["model"]["text"]["word_embedding_dim"],
        lstm_hidden_dim=config["model"]["text"]["lstm_hidden_dim"],
        lstm_num_layers=config["model"]["text"]["lstm_num_layers"],
        lstm_bidirectional=config["model"]["text"]["lstm_bidirectional"],
        projection_dim=config["model"]["image"]["projection_dim"],
        temperature_init=config["model"]["temperature_init"]
    )
    model = model.to(device)
    
    loss_fn = CLIPLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    epochs = config["training"]["epochs"]
    lr_scheduler_type = config["training"]["lr_scheduler"]
    
    if lr_scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif lr_scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config["training"]["step_size"], 
            gamma=config["training"]["gamma"]
        )
    else:
        scheduler = None
        
    # 6. Execute Training Loop
    best_val_loss = float("inf")
    
    print("\nStarting Pretraining Pipeline...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            log_interval=config["training"]["log_interval"]
        )
        
        val_results = evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device
        )
        
        val_loss = val_results["val_loss"]
        metrics = val_results["metrics"]
        
        # Step the scheduler if it's epoch-based
        if scheduler is not None:
            scheduler.step()
            
        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Retrieval Metrics (Val Dataset):")
        print(f"  Image-to-Text (I2T) -> Recall@1: {metrics['i2t_r1']:.2f}% | Recall@5: {metrics['i2t_r5']:.2f}% | Recall@10: {metrics['i2t_r10']:.2f}%")
        print(f"  Text-to-Image (T2I) -> Recall@1: {metrics['t2i_r1']:.2f}% | Recall@5: {metrics['t2i_r5']:.2f}% | Recall@10: {metrics['t2i_r10']:.2f}%")
        print("-" * 30 + "\n")
        
        # 7. Checkpoint Saving Logic
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_word2idx": vocab.word2idx,
            "vocab_idx2word": vocab.idx2word,
            "config": config,
            "val_metrics": metrics,
            "val_loss": val_loss
        }
        
        # Save latest model weights
        latest_path = os.path.join(checkpoint_dir, "latest_clip_model.pt")
        torch.save(checkpoint, latest_path)
        
        # Save best model weights based on validation loss improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(checkpoint_dir, "best_clip_model.pt")
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] Validation Loss improved. Saved best model to: {best_path}")

    print("CLIP pretraining completed successfully.")

if __name__ == "__main__":
    main()
