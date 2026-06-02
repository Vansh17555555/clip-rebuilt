import os
import torch
from PIL import Image
from typing import List, Tuple, Dict, Any
from clip_from_scratch.data.vocabulary import Vocabulary
from clip_from_scratch.data.tokenizer import Tokenizer
from clip_from_scratch.data.dataset import FlickrDataset
from clip_from_scratch.models.clip_model import CLIPModel

class ImageToTextRetriever:
    """
    ImageToTextRetriever handles image queries by searching a database of text
    descriptions, ranking their similarity using a pretrained custom CLIP model,
    and returning the most relevant captions (Image -> Text cross-modal retrieval).
    """
    def __init__(self, checkpoint_path: str):
        """
        Args:
            checkpoint_path: Path to the saved CLIP model checkpoint .pt file.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load checkpoint on CPU first to prevent CUDA out-of-memory
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
            
        print(f"Loading checkpoint from: {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # 2. Recreate Vocabulary and Tokenizer
        self.vocab = Vocabulary()
        self.vocab.word2idx = checkpoint["vocab_word2idx"]
        self.vocab.idx2word = {int(k): v for k, v in checkpoint["vocab_idx2word"].items()}
        self.tokenizer = Tokenizer(self.vocab)
        
        # 3. Instantiate and load Model
        config = checkpoint["config"]
        self.max_seq_len = config["dataset"]["max_seq_len"]
        
        self.model = CLIPModel(
            vocab_size=len(self.vocab),
            pad_idx=self.vocab.word2idx[self.vocab.pad_token],
            image_pretrained=False,
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
        self.captions: List[str] = []
        self.cached_text_embeddings: torch.Tensor = torch.empty(0)
        self.image_transform = FlickrDataset.get_default_transform(is_train=False)

    @torch.no_grad()
    def build_text_index(self, captions_list: List[str]) -> None:
        """
        Encodes all captions in the given list, caching their embeddings
        to make search queries extremely fast.
        """
        print(f"Building retrieval index for {len(captions_list)} captions...")
        self.captions = captions_list
        embeddings_list = []
        
        # Batch size for text encoding to avoid large GPU memory overhead
        batch_size = 128
        for i in range(0, len(captions_list), batch_size):
            batch_caps = captions_list[i : i + batch_size]
            encoded_batch = []
            
            for cap in batch_caps:
                encoded_batch.append(self.tokenizer.encode(cap, max_seq_len=self.max_seq_len))
                
            tokens_tensor = torch.tensor(encoded_batch, dtype=torch.long, device=self.device)
            
            # Encode text
            txt_embed = self.model.text_encoder(tokens_tensor)
            embeddings_list.append(txt_embed.cpu())
            
        self.cached_text_embeddings = torch.cat(embeddings_list, dim=0)
        print("Text index built successfully.")

    @torch.no_grad()
    def search(self, image_path_or_pil: Any, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Searches the caption database for the top-k texts that describe the input image.
        
        Args:
            image_path_or_pil: Path string to image, or opened PIL Image object.
            top_k: Number of matches to return.
            
        Returns:
            A list of tuples: (caption_string, similarity_score) sorted by score.
        """
        if self.cached_text_embeddings.numel() == 0:
            raise ValueError("Text index has not been built. Call build_text_index() first.")
            
        # 1. Load and transform image
        if isinstance(image_path_or_pil, torch.Tensor):
            if image_path_or_pil.ndim == 3:
                image_tensor = image_path_or_pil.unsqueeze(0).to(self.device)
            else:
                image_tensor = image_path_or_pil.to(self.device)
        elif isinstance(image_path_or_pil, str):
            if os.path.exists(image_path_or_pil):
                image = Image.open(image_path_or_pil).convert("RGB")
                image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            else:
                raise FileNotFoundError(f"Image not found at {image_path_or_pil}")
        else:
            image = image_path_or_pil.convert("RGB")
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        
        # 2. Extract image embedding -> [1, projection_dim]
        image_embedding = self.model.image_encoder(image_tensor).cpu()
        
        # 3. Compute cosine similarities against all cached text embeddings
        # similarities = [N_captions]
        similarities = torch.matmul(self.cached_text_embeddings, image_embedding.T).squeeze(1)
        
        # 4. Retrieve top-k results
        k = min(top_k, len(self.captions))
        scores, indices = torch.topk(similarities, k=k, largest=True)
        
        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            results.append((self.captions[idx], score))
            
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
        # Instantiate retriever
        retriever = ImageToTextRetriever(checkpoint_path=checkpoint_file)
        
        # Define candidate captions to search from
        candidate_captions = [
            "a happy dog running across green fields.",
            "a white cat sleeping on top of the red sofa.",
            "a young boy climbing a tall tree in the playground.",
            "a red sports car speeding down an empty road.",
            "people walking down a busy city street in rain.",
        ]
        retriever.build_text_index(candidate_captions)
        
        # Search with dummy image (will fallback to drawing a colored box)
        dummy_image = Image.new("RGB", (224, 224), color=(34, 139, 34)) # forest green background
        
        print("\nSearching captions matching a green canvas image...")
        results = retriever.search(dummy_image, top_k=3)
        for i, (caption, score) in enumerate(results):
            print(f"Top {i+1}: '{caption}' (Similarity: {score:.4f})")
    else:
        print("[Info] No checkpoint found. Skipping retriever local test. This script is fully ready for deployment.")

