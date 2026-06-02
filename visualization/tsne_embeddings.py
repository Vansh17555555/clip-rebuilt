import os
import torch
import numpy as np
from typing import List, Tuple
from torch.utils.data import DataLoader

# Try to import visualization libraries safely
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from clip_from_scratch.data.vocabulary import Vocabulary
from clip_from_scratch.data.tokenizer import Tokenizer
from clip_from_scratch.data.dataset import FlickrDataset, collate_fn
from clip_from_scratch.models.clip_model import CLIPModel

@torch.no_grad()
def extract_embeddings(checkpoint_path: str, max_samples: int = 150) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Loads a pretrained model and extracts image and text embeddings for validation samples.
    
    Returns:
        image_embeds: Numpy array of shape [N, projection_dim]
        text_embeds: Numpy array of shape [N, projection_dim]
        raw_captions: List of N raw caption strings
        img_names: List of N image filenames
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    
    # Rebuild Vocab & Tokenizer
    vocab = Vocabulary()
    vocab.word2idx = checkpoint["vocab_word2idx"]
    vocab.idx2word = {int(k): v for k, v in checkpoint["vocab_idx2word"].items()}
    tokenizer = Tokenizer(vocab)
    
    # Load dataset
    dataset = FlickrDataset(
        image_dir=config["dataset"]["image_dir"],
        caption_file=config["dataset"]["caption_file"],
        tokenizer=tokenizer,
        max_seq_len=config["dataset"]["max_seq_len"],
        transform=FlickrDataset.get_default_transform(is_train=False)
    )
    
    # Cap samples to keep visualization clean and fast
    if len(dataset) > max_samples:
        # Keep it deterministic
        indices = list(range(max_samples))
        dataset.data_pairs = [dataset.data_pairs[i] for i in indices]
        
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Rebuild model
    model = CLIPModel(
        vocab_size=len(vocab),
        pad_idx=vocab.word2idx[vocab.pad_token],
        image_pretrained=False,
        feature_dim=config["model"]["image"]["feature_dim"],
        word_embedding_dim=config["model"]["text"]["word_embedding_dim"],
        lstm_hidden_dim=config["model"]["text"]["lstm_hidden_dim"],
        lstm_num_layers=config["model"]["text"]["lstm_num_layers"],
        lstm_bidirectional=config["model"]["text"]["lstm_bidirectional"],
        projection_dim=config["model"]["image"]["projection_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    try:
        model = model.to(device)
    except RuntimeError as e:
        if "CUDA" in str(e) or "device" in str(e):
            print(f"[Warning] GPU ({device}) is busy or unavailable. Error: {e}")
            print("Falling back to CPU mode...")
            device = torch.device("cpu")
            model = model.to(device)
        else:
            raise e
    model.eval()
    
    # Clean up checkpoint from memory
    del checkpoint
    
    img_embeds_list = []
    txt_embeds_list = []
    all_captions = []
    all_img_names = []
    
    for batch in loader:
        images = batch["image"].to(device)
        captions = batch["caption"].to(device)
        
        outputs = model(images, captions)
        
        img_embeds_list.append(outputs["image_embeddings"].cpu().numpy())
        txt_embeds_list.append(outputs["text_embeddings"].cpu().numpy())
        
        all_captions.extend(batch["raw_caption"])
        all_img_names.extend(batch["image_name"])
        
    return (
        np.concatenate(img_embeds_list, axis=0),
        np.concatenate(txt_embeds_list, axis=0),
        all_captions,
        all_img_names
    )

def plot_and_save_embeddings(image_embeds: np.ndarray, 
                             text_embeds: np.ndarray, 
                             captions: List[str], 
                             output_dir: str = "./visualizations",
                             method: str = "tsne") -> str:
    """
    Reduces embeddings to 2D using t-SNE or UMAP and plots them,
    drawing connecting lines between matching image-text pairs.
    """
    if not HAS_MATPLOTLIB:
        print("[Error] matplotlib is required for plotting embeddings.")
        return ""
    if not HAS_SKLEARN:
        print("[Error] scikit-learn is required for t-SNE / PCA reduction.")
        return ""
        
    os.makedirs(output_dir, exist_ok=True)
    num_samples = image_embeds.shape[0]
    
    # 1. Stack all embeddings to project together
    # Shape: [2 * N, projection_dim]
    all_embeds = np.concatenate([image_embeds, text_embeds], axis=0)
    
    # 2. Perform Dimensionality Reduction
    print(f"Reducing dimensions of {all_embeds.shape[0]} embeddings using {method.upper()}...")
    if method.lower() == "tsne":
        # Perplexity should be adjusted for small sizes
        perp = min(30, max(5, num_samples // 4))
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
        coords = reducer.fit_transform(all_embeds)
    elif method.lower() == "umap":
        if HAS_UMAP:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            coords = reducer.fit_transform(all_embeds)
        else:
            print("[Warning] UMAP not installed. Falling back to t-SNE.")
            perp = min(30, max(5, num_samples // 4))
            reducer = TSNE(n_components=2, perplexity=perp, random_state=42)
            coords = reducer.fit_transform(all_embeds)
            method = "tsne"
    else:
        print("[Info] Method not recognized. Using PCA fallback.")
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(all_embeds)
        method = "pca"
        
    # Split coordinates back into image and text
    img_coords = coords[:num_samples]
    txt_coords = coords[num_samples:]
    
    # 3. Create the Premium-style dark grid plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    
    # Draw faint connecting lines between matching text and image embeddings
    # This visualizes the contrastive alignment pull!
    for i in range(num_samples):
        ax.plot(
            [img_coords[i, 0], txt_coords[i, 0]],
            [img_coords[i, 1], txt_coords[i, 1]],
            color="white", alpha=0.12, linestyle="--", linewidth=0.8
        )
        
    # Plot Image nodes
    img_scatter = ax.scatter(
        img_coords[:, 0], img_coords[:, 1],
        color="#FF007F", label="Image Embeddings", alpha=0.85, edgecolors='black', s=55, marker="o"
    )
    
    # Plot Text nodes
    txt_scatter = ax.scatter(
        txt_coords[:, 0], txt_coords[:, 1],
        color="#00E5FF", label="Text Embeddings", alpha=0.85, edgecolors='black', s=55, marker="^"
    )
    
    # Annotate a few samples to make the chart readable
    annotation_cap = min(5, num_samples)
    for i in range(annotation_cap):
        cleaned_caption = captions[i][:25] + "..." if len(captions[i]) > 25 else captions[i]
        # Text annotation
        ax.annotate(
            f"T{i}: {cleaned_caption}",
            (txt_coords[i, 0], txt_coords[i, 1]),
            xytext=(5, 2), textcoords="offset points", fontsize=8, color="#B3F3FD", alpha=0.9
        )
        # Image annotation
        ax.annotate(
            f"Img{i}",
            (img_coords[i, 0], img_coords[i, 1]),
            xytext=(5, -6), textcoords="offset points", fontsize=8, color="#FFB3D9", alpha=0.9
        )

    ax.set_title(f"CLIP Shared Latent Space Projection ({method.upper()})", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", frameon=True, facecolor="#1e1e1e", edgecolor="#333333")
    ax.grid(True, linestyle=":", alpha=0.2, color="#444444")
    
    # Premium detailing margins
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"clip_embeddings_{method}.png")
    plt.savefig(save_path, facecolor="#121212", edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Success! Visualization plot saved as image: {save_path}")
    return save_path

if __name__ == "__main__":
    # Robust path selection supporting Local Windows, Colab Linux, root level, and relative directories
    possible_paths = [
        "c:/Users/Vansh/clip/best_clip_model.pt",
        "c:/Users/Vansh/clip/clip_from_scratch/checkpoints/best_clip_model.pt",
        "/content/clip_from_scratch/checkpoints/best_clip_model.pt",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "best_clip_model.pt"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "best_clip_model.pt")
    ]
    
    checkpoint_file = None
    for p in possible_paths:
        if os.path.exists(p):
            checkpoint_file = p
            break

        
    if checkpoint_file is not None and os.path.exists(checkpoint_file):
        try:
            img_e, txt_e, caps, names = extract_embeddings(checkpoint_file, max_samples=40)
            
            # Save t-SNE plot
            plot_and_save_embeddings(img_e, txt_e, caps, method="tsne")
            
            # Save UMAP plot
            plot_and_save_embeddings(img_e, txt_e, caps, method="umap")
        except Exception as e:
            print(f"[Error] Failed to generate visualizations: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[Info] No checkpoint found at {checkpoint_file}. Skipping visualizer test. Script is fully prepared for evaluation.")
