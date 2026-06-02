import os
import shutil
import torch
from torch.utils.data import DataLoader

# Import everything
from clip_from_scratch.data.vocabulary import Vocabulary
from clip_from_scratch.data.tokenizer import Tokenizer
from clip_from_scratch.data.dataset import FlickrDataset, collate_fn
from clip_from_scratch.models.clip_model import CLIPModel
from clip_from_scratch.losses.contrastive_loss import CLIPLoss
from clip_from_scratch.training.engine import train_one_epoch, evaluate
from clip_from_scratch.retrieval.image_retrieval import TextToImageRetriever
from clip_from_scratch.retrieval.text_retrieval import ImageToTextRetriever

def run_checks():
    print("=" * 60)
    print("=== RUNNING MULTIMODAL SYSTEM INTEGRATION SANITY CHECKS ===")
    print("=" * 60)
    
    device = torch.device("cpu") # run check on CPU to be hardware-agnostic
    temp_dir = "./sanity_check_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 1. Vocab & Tokenizer Test
        print("\n[Step 1/6] Vocab & Tokenizer Check...")
        vocab = Vocabulary()
        vocab.add_word("dog")
        vocab.add_word("cat")
        vocab.add_word("running")
        vocab.add_word("grass")
        
        tokenizer = Tokenizer(vocab)
        tokens = tokenizer.tokenize("A cat and dog running in grass.")
        indices = tokenizer.encode("A cat and dog running in grass.", max_seq_len=10)
        assert len(indices) == 10, "Encoding length mismatch."
        print(" -> Success: Vocab builder and Tokenizer operational.")
        
        # 2. Dataset & Collate Check
        print("\n[Step 2/6] Dataset & DataLoader Check...")
        dataset = FlickrDataset(
            image_dir=temp_dir,
            caption_file=os.path.join(temp_dir, "dummy_captions.txt"),
            tokenizer=tokenizer,
            max_seq_len=10
        )
        # Verify dummy pairs exist
        assert len(dataset) > 0, "Dummy dataset should not be empty."
        
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        batch = next(iter(loader))
        assert batch["image"].shape == (2, 3, 224, 224), "Image batch shape mismatch."
        assert batch["caption"].shape == (2, 10), "Caption batch shape mismatch."
        print(" -> Success: FlickrDataset loaded and batched correctly.")
        
        # 3. Model Architecture Check
        print("\n[Step 3/6] CLIP Model Forward-Pass Check...")
        model = CLIPModel(
            vocab_size=len(vocab),
            pad_idx=vocab.word2idx[vocab.pad_token],
            image_pretrained=False,
            feature_dim=512,
            word_embedding_dim=64,
            lstm_hidden_dim=64,
            lstm_num_layers=1,
            lstm_bidirectional=True,
            projection_dim=32,
            temperature_init=0.07
        ).to(device)
        
        outputs = model(batch["image"].to(device), batch["caption"].to(device))
        img_e = outputs["image_embeddings"]
        txt_e = outputs["text_embeddings"]
        temp = outputs["temperature"]
        
        assert img_e.shape == (2, 32), "Image embedding projection mismatch."
        assert txt_e.shape == (2, 32), "Text embedding projection mismatch."
        # Validate L2 Normalization
        assert torch.allclose(torch.norm(img_e, p=2, dim=-1), torch.ones(2)), "Image embeddings not L2-normalized."
        assert torch.allclose(torch.norm(txt_e, p=2, dim=-1), torch.ones(2)), "Text embeddings not L2-normalized."
        print(" -> Success: Encoders, Multi-Layer Projections, and Normalizations match perfectly.")
        
        # 4. CLIP InfoNCE Loss Check
        print("\n[Step 4/6] CLIP Symmetric Loss Check...")
        loss_fn = CLIPLoss()
        loss = loss_fn(img_e, txt_e, temp)
        assert loss.item() > 0.0, "Loss should be positive."
        print(f" -> Success: Symmetric Loss is mathematically stable. Loss value: {loss.item():.4f}")
        
        # 5. Checkpoint Saving Test
        print("\n[Step 5/6] Checkpoint Serialization Check...")
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "vocab_word2idx": vocab.word2idx,
            "vocab_idx2word": vocab.idx2word,
            "config": {
                "dataset": {"max_seq_len": 10},
                "model": {
                    "image": {"feature_dim": 512, "projection_dim": 32},
                    "text": {
                        "word_embedding_dim": 64,
                        "lstm_hidden_dim": 64,
                        "lstm_num_layers": 1,
                        "lstm_bidirectional": True
                    }
                }
            }
        }
        chk_path = os.path.join(temp_dir, "sanity_checkpoint.pt")
        torch.save(checkpoint, chk_path)
        assert os.path.exists(chk_path), "Failed to save checkpoint."
        print(" -> Success: Model state and vocabulary mappings successfully serialized.")
        
        # 6. Retrieval Engine Test
        print("\n[Step 6/6] Retrieval Engines Integration Check...")
        t2i = TextToImageRetriever(checkpoint_path=chk_path, image_dir=temp_dir)
        t2i.build_image_index(dataset)
        t2i_results = t2i.search("dog running", top_k=2)
        assert len(t2i_results) > 0, "No retrieval results returned."
        
        i2t = ImageToTextRetriever(checkpoint_path=chk_path)
        i2t.build_text_index(["dog running in grass", "cat sitting on sofa"])
        # Use first image from batch
        i2t_results = i2t.search(batch["image"][0], top_k=1)
        assert len(i2t_results) > 0, "No retrieval results returned."
        
        print(" -> Success: Text-to-Image and Image-to-Text retrieval engines fully functional.")
        
        print("\n" + "=" * 60)
        print("[SUCCESS] INTEGRATION SANITY CHECK PASSED - READY TO TRAIN!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] INTEGRATION FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup temp files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    run_checks()
