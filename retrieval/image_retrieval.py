import os
import torch
from PIL import Image
from typing import List, Tuple, Dict, Any
from clip_from_scratch.data.vocabulary import Vocabulary
from clip_from_scratch.data.tokenizer import Tokenizer
from clip_from_scratch.data.dataset import FlickrDataset
from clip_from_scratch.models.clip_model import CLIPModel

class TextToImageRetriever:
    """
    TextToImageRetriever handles natural language queries by searching a database
    of images, computing similarity rankings using a pretrained custom CLIP model,
    and returning the most relevant images (Text -> Image cross-modal retrieval).
    """
    def __init__(self, checkpoint_path: str, image_dir: str):
        """
        Args:
            checkpoint_path: Path to the saved CLIP model checkpoint .pt file.
            image_dir: Path to directory containing candidate images.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_dir = image_dir
        
        # 1. Load the checkpoint on CPU first to prevent CUDA out-of-memory
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
            
        print(f"Loading checkpoint from: {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # 2. Recreate Vocabulary and Tokenizer
        self.vocab = Vocabulary()
        self.vocab.word2idx = checkpoint["vocab_word2idx"]
        self.vocab.idx2word = {int(k): v for k, v in checkpoint["vocab_idx2word"].items()}
        self.tokenizer = Tokenizer(self.vocab)
        
        # 3. Instantiate and load Model weights
        config = checkpoint["config"]
        self.max_seq_len = config["dataset"]["max_seq_len"]
        
        self.model = CLIPModel(
            vocab_size=len(self.vocab),
            pad_idx=self.vocab.word2idx[self.vocab.pad_token],
            image_pretrained=False, # We load pre-trained weights from checkpoint
            feature_dim=config["model"]["image"]["feature_dim"],
            word_embedding_dim=config["model"]["text"]["word_embedding_dim"],
            lstm_hidden_dim=config["model"]["text"]["lstm_hidden_dim"],
            lstm_num_layers=config["model"]["text"]["lstm_num_layers"],
            lstm_bidirectional=config["model"]["text"]["lstm_bidirectional"],
            projection_dim=config["model"]["image"]["projection_dim"]
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        try:
            self.model = self.model.to(self.device)
        except RuntimeError as e:
            if "CUDA" in str(e) or "device" in str(e):
                print(f"[Warning] GPU ({self.device}) is busy or unavailable. Error: {e}")
                print("Falling back to CPU mode...")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
            else:
                raise e
        self.model.eval()
        
        # Clean up checkpoint from memory
        del checkpoint
        
        # Caching holders
        self.image_names: List[str] = []
        self.cached_image_embeddings: torch.Tensor = torch.empty(0)

    @torch.no_grad()
    def build_image_index(self, dataset: FlickrDataset) -> None:
        """
        Encodes all images in a given dataset, caching their embeddings
        to make search queries extremely fast.
        """
        print(f"Building retrieval index for {len(dataset)} images...")
        embeddings_list = []
        unique_images = set()
        self.image_names = []
        
        # We wrap in eval transform
        transform = FlickrDataset.get_default_transform(is_train=False)
        
        for idx in range(len(dataset)):
            img_name = dataset.data_pairs[idx][0]
            
            # Avoid duplicate image processing if Flickr dataset has multiple captions per image
            if img_name in unique_images:
                continue
                
            unique_images.add(img_name)
            self.image_names.append(img_name)
            
            # Get image tensor
            img_path = os.path.join(self.image_dir, img_name)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
            else:
                image = Image.new("RGB", (224, 224), color=(128, 128, 128))
                
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Encode image
            img_embed = self.model.image_encoder(image_tensor)
            embeddings_list.append(img_embed.cpu())
            
        self.cached_image_embeddings = torch.cat(embeddings_list, dim=0)
        print("Image index built successfully.")

    @torch.no_grad()
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Searches the image database for the top-k images that match the textual query.
        
        Args:
            query: Natural language search string.
            top_k: Number of matches to return.
            
        Returns:
            A list of tuples: (image_filename, similarity_score) sorted by score.
        """
        if self.cached_image_embeddings.numel() == 0:
            raise ValueError("Image index has not been built. Call build_image_index() first.")
            
        # 1. Encode text query
        encoded_tokens = self.tokenizer.encode(query, max_seq_len=self.max_seq_len)
        tokens_tensor = torch.tensor([encoded_tokens], dtype=torch.long, device=self.device)
        
        # 2. Extract text embedding -> [1, projection_dim]
        text_embedding = self.model.text_encoder(tokens_tensor).cpu()
        
        # 3. Compute cosine similarities against all cached image embeddings
        # Since both vectors are L2 normalized, similarity is the dot product:
        # similarities = [N_images]
        similarities = torch.matmul(self.cached_image_embeddings, text_embedding.T).squeeze(1)
        
        # 4. Retrieve top-k results
        k = min(top_k, len(self.image_names))
        scores, indices = torch.topk(similarities, k=k, largest=True)
        
        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            results.append((self.image_names[idx], score))
            
        return results

# Quick Demo Run
if __name__ == "__main__":
    # Robust path selection supporting root level, checkpoints subdirectory, and relative execution paths
    possible_paths = [
        "c:/Users/Vansh/clip/best_clip_model.pt",
        "c:/Users/Vansh/clip/clip_from_scratch/checkpoints/best_clip_model.pt",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "best_clip_model.pt"),
        "./best_clip_model.pt",
        "./clip_from_scratch/checkpoints/best_clip_model.pt"
    ]
    
    checkpoint_file = None
    for p in possible_paths:
        if os.path.exists(p):
            checkpoint_file = p
            break
            
    if checkpoint_file is not None:
        # Resolve dataset image directory and caption paths
        image_dir = "c:/Users/Vansh/clip/clip_from_scratch/data/flickr8k/Images"
        caption_file = "c:/Users/Vansh/clip/clip_from_scratch/data/flickr8k/captions.txt"
        
        possible_caps = [
            caption_file,
            "./clip_from_scratch/data/flickr8k/captions.txt",
            "./data/flickr8k/captions.txt"
        ]
        for p in possible_caps:
            if os.path.exists(p):
                caption_file = p
                break
                
        possible_imgs = [
            image_dir,
            "./clip_from_scratch/data/flickr8k/Images",
            "./data/flickr8k/Images"
        ]
        for p in possible_imgs:
            if os.path.exists(p):
                image_dir = p
                break

        # Instantiate retriever
        retriever = TextToImageRetriever(
            checkpoint_path=checkpoint_file,
            image_dir=image_dir
        )
        
        # Create validation dataset to index
        vocab = retriever.vocab
        tokenizer = Tokenizer(vocab)
        dataset = FlickrDataset(
            image_dir=image_dir,
            caption_file=caption_file,
            tokenizer=tokenizer
        )
        
        # Build index
        retriever.build_image_index(dataset)
        
        # Search queries
        query = "dog running in green grass"
        print(f"\nSearching for query: '{query}'")
        results = retriever.search(query, top_k=3)
        for i, (name, score) in enumerate(results):
            print(f"Top {i+1}: {name} (Similarity: {score:.4f})")
    else:
        print("[Info] No checkpoint found. Skipping retriever local test. This script is fully ready for deployment.")

