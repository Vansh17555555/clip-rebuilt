import json
from collections import Counter
import os
from typing import List, Dict, Union

class Vocabulary:
    """
    Vocabulary class that manages word-to-index and index-to-word mappings,
    vital for encoding text descriptions into token tensors for the text encoder.
    """
    def __init__(self, pad_token: str = "<pad>", unk_token: str = "<unk>", 
                 start_token: str = "<start>", end_token: str = "<end>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token

        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        
        # Add special tokens
        self.add_word(self.pad_token)  # Typically index 0
        self.add_word(self.unk_token)  # Typically index 1
        self.add_word(self.start_token)  # Typically index 2
        self.add_word(self.end_token)  # Typically index 3

    def add_word(self, word: str) -> int:
        """Adds a word to the vocabulary if it doesn't already exist."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            return idx
        return self.word2idx[word]

    def get_idx(self, word: str) -> int:
        """Returns the index of a word, falling back to the <unk> token if missing."""
        return self.word2idx.get(word, self.word2idx[self.unk_token])

    def get_word(self, idx: int) -> str:
        """Returns the word corresponding to a given index, falling back to <unk>."""
        return self.idx2word.get(idx, self.unk_token)

    def __len__(self) -> int:
        return len(self.word2idx)

    def build_vocabulary(self, sentences: List[List[str]], min_freq: int = 2) -> None:
        """
        Builds the vocabulary mapping from a list of tokenized sentences,
        retaining only words that meet a minimum frequency threshold.
        
        Args:
            sentences: List of token lists, e.g., [['a', 'dog', 'runs'], ['two', 'cats']]
            min_freq: Minimum frequency of a word to be added to the vocabulary.
        """
        counter = Counter()
        for sentence in sentences:
            counter.update(sentence)

        # Filter out words with frequency below threshold
        for word, count in counter.items():
            if count >= min_freq:
                self.add_word(word)

    def save(self, file_path: str) -> None:
        """Saves the vocabulary mappings as a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "word2idx": self.word2idx,
                "idx2word": {str(k): v for k, v in self.idx2word.items()}
            }, f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: str) -> 'Vocabulary':
        """Loads a vocabulary mappings from a saved JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls()
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        return vocab

# Quick Sanity Test
if __name__ == "__main__":
    test_sentences = [
        ["a", "brown", "dog", "running", "on", "the", "grass"],
        ["a", "white", "dog", "playing", "with", "a", "ball"],
        ["a", "cat", "sleeping", "on", "the", "sofa"]
    ]
    vocab = Vocabulary()
    vocab.build_vocabulary(test_sentences, min_freq=2)
    print(f"Vocabulary size after filtering: {len(vocab)}")
    print(f"Index of 'dog': {vocab.get_idx('dog')}")
    print(f"Index of 'sleeping': {vocab.get_idx('sleeping')} (should be <unk> if freq < 2)")
    print(f"Word at index 0: {vocab.get_word(0)} (should be <pad>)")
