import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Dict, Tuple, Optional, Union
from clip_from_scratch.data.tokenizer import Tokenizer

class FlickrDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Flickr8k (or similar) image-caption pairs.
    Each item returns the augmented image tensor, the encoded token indices of the caption,
    and metadata for evaluation (raw caption, image filename).
    """
    def __init__(self, 
                 image_dir: str, 
                 caption_file: str, 
                 tokenizer: Tokenizer, 
                 max_seq_len: int = 32, 
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            image_dir: Directory containing Flickr images.
            caption_file: CSV or text file containing captions (format: image,caption).
            tokenizer: An initialized Tokenizer instance.
            max_seq_len: Maximum token length for captions.
            transform: PyTorch image transformation pipeline.
        """
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = transform or self.get_default_transform(is_train=False)

        # Parse captions file. Standard Flickr8k captions file layout has a header: image,caption
        # and subsequent lines are: image_name,caption
        self.data_pairs: List[Tuple[str, str]] = []
        
        if not os.path.exists(caption_file):
            print(f"[Warning] Caption file not found at: {caption_file}. Creating dataset with synthetic dummy data.")
            self._create_dummy_data()
        else:
            self._load_dataset(caption_file)

    def _load_dataset(self, caption_file: str) -> None:
        """Loads and parses the caption file into a list of (image_filename, caption_string) tuples."""
        with open(caption_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        header_skipped = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip CSV header if present
            if not header_skipped and ("image,caption" in line.lower() or "image_name" in line.lower()):
                header_skipped = True
                continue
            
            # Some caption formats might use tab or commas. We support commas and split on the first one only.
            if ',' in line:
                parts = line.split(',', 1)
            elif '\t' in line:
                parts = line.split('\t', 1)
            else:
                continue
                
            if len(parts) == 2:
                img_name = parts[0].strip()
                caption = parts[1].strip()
                # Remove quotes if they exist around the caption
                if caption.startswith('"') and caption.endswith('"'):
                    caption = caption[1:-1]
                
                self.data_pairs.append((img_name, caption))
                
        print(f"Loaded {len(self.data_pairs)} image-caption pairs from {caption_file}.")

    def _create_dummy_data(self) -> None:
        """Creates dummy data for testing/debugging when Flickr8k is not local."""
        print("[Info] Creating synthetic dummy dataset.")
        # Create 5 synthetic image filenames and associated captions
        dummy_captions = [
            ("img1.jpg", "A black dog running in the green grass."),
            ("img1.jpg", "A playful dog chasing a ball on the lawn."),
            ("img2.jpg", "A small white cat sitting on top of a red sofa."),
            ("img3.jpg", "A young boy climbing a tall tree in the park."),
            ("img4.jpg", "A red car parked on the side of a busy street."),
        ]
        self.data_pairs = dummy_captions

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """
        Returns:
            image_tensor: Normalized image tensor [3, 224, 224]
            caption_tensor: Padded token indices tensor [max_seq_len]
            raw_caption: The original string caption
            img_name: The image filename
        """
        img_name, raw_caption = self.data_pairs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image; fallback to dummy canvas if file is missing
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                # Fallback on corrupted files
                image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        else:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Apply transformations (augmentations, scaling, norm)
        image_tensor = self.transform(image)
        
        # Tokenize and encode caption
        encoded_caption = self.tokenizer.encode(raw_caption, max_seq_len=self.max_seq_len)
        caption_tensor = torch.tensor(encoded_caption, dtype=torch.long)
        
        return image_tensor, caption_tensor, raw_caption, img_name

    @staticmethod
    def get_default_transform(is_train: bool = True) -> transforms.Compose:
        """
        Defines standard ImageNet preprocessing transforms.
        Augmentations are applied during training to enhance generalization.
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if is_train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, str, str]]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    Custom collate function for DataLoader.
    Organizes tensors into batches and keeps textual metadata as clean lists.
    """
    images, captions, raw_captions, img_names = zip(*batch)
    
    # Stack image tensors [B, 3, 224, 224]
    images_batch = torch.stack(images, dim=0)
    # Stack caption tensors [B, max_seq_len]
    captions_batch = torch.stack(captions, dim=0)
    
    return {
        "image": images_batch,
        "caption": captions_batch,
        "raw_caption": list(raw_captions),
        "image_name": list(img_names)
    }

# Quick Sanity Test
if __name__ == "__main__":
    from clip_from_scratch.data.vocabulary import Vocabulary
    
    # 1. Build small vocab
    vocab = Vocabulary()
    vocab.add_word("black")
    vocab.add_word("dog")
    vocab.add_word("running")
    vocab.add_word("in")
    vocab.add_word("the")
    vocab.add_word("green")
    vocab.add_word("grass")
    
    # 2. Setup tokenizer
    tokenizer = Tokenizer(vocab)
    
    # 3. Initialize dataset (will trigger synthetic dummy data because no files exist yet)
    dataset = FlickrDataset(
        image_dir="./dummy_images",
        caption_file="./dummy_captions.txt",
        tokenizer=tokenizer,
        max_seq_len=12,
        transform=FlickrDataset.get_default_transform(is_train=True)
    )
    
    print(f"Dataset length: {len(dataset)}")
    img, cap, raw, name = dataset[0]
    print(f"Image tensor shape: {img.shape}")
    print(f"Caption tensor: {cap}")
    print(f"Raw caption: '{raw}'")
    print(f"Image filename: '{name}'")
    
    # 4. Try DataLoader with custom collate function
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"DataLoader batch - Image shape: {batch['image'].shape}")
    print(f"DataLoader batch - Caption shape: {batch['caption'].shape}")
    print(f"DataLoader batch - Image names: {batch['image_name']}")
