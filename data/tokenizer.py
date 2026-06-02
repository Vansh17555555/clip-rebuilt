import re
from typing import List
from clip_from_scratch.data.vocabulary import Vocabulary

class Tokenizer:
    """
    Word-level tokenizer that cleans, normalizes, and converts natural language
    captions into standardized lists of text tokens, and subsequently into index sequences
    ready to be embedded by the Text Encoder.
    """
    def __init__(self, vocabulary: Vocabulary):
        self.vocab = vocabulary

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans input string by lowercasing and removing punctuation.
        This ensures words like 'dog.' and 'dog' map to the same vocabulary key.
        """
        # Convert to lowercase
        text = text.lower().strip()
        # Remove punctuation, keeping alphanumeric characters and spaces
        text = re.sub(r"[^\w\s]", "", text)
        # Collapse multiple spaces into one
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Cleans and tokenizes text into a list of word tokens.
        """
        cleaned = self.clean_text(text)
        return [token for token in cleaned.split(" ") if token]

    def encode(self, text: str, max_seq_len: int = 32) -> List[int]:
        """
        Encodes raw text into a sequence of token indices of fixed length `max_seq_len`.
        Prepend '<start>' and append '<end>'. Pad with '<pad>' or truncate.
        
        Args:
            text: Raw input caption.
            max_seq_len: Target sequence length.
            
        Returns:
            A list of integers representing token indices.
        """
        tokens = self.tokenize(text)
        
        # Determine available tokens capacity for content (subtract start and end)
        content_capacity = max_seq_len - 2
        if len(tokens) > content_capacity:
            tokens = tokens[:content_capacity]
            
        # Reassemble token stream with start/end boundary markers
        full_tokens = [self.vocab.start_token] + tokens + [self.vocab.end_token]
        
        # Pad to max_seq_len if needed
        padding_needed = max_seq_len - len(full_tokens)
        if padding_needed > 0:
            full_tokens += [self.vocab.pad_token] * padding_needed
            
        # Map words to indices
        indices = [self.vocab.get_idx(token) for token in full_tokens]
        return indices

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """
        Decodes a list of token indices back into a readable string representation.
        
        Args:
            indices: List of integer indices.
            skip_special: If true, filters out <pad>, <start>, <end>, and <unk>.
        """
        words = []
        for idx in indices:
            word = self.vocab.get_word(idx)
            if skip_special:
                if word in [self.vocab.pad_token, self.vocab.start_token, self.vocab.end_token]:
                    continue
            words.append(word)
        return " ".join(words)

# Quick Sanity Test
if __name__ == "__main__":
    vocab = Vocabulary()
    vocab.add_word("dog")
    vocab.add_word("runs")
    vocab.add_word("fast")
    
    tokenizer = Tokenizer(vocab)
    raw_text = "A dog, runs fast!"
    tokens = tokenizer.tokenize(raw_text)
    print(f"Tokenized: {tokens}")  # ['a', 'dog', 'runs', 'fast']
    
    encoded = tokenizer.encode(raw_text, max_seq_len=8)
    print(f"Encoded indices (max_seq_len=8): {encoded}")
    # Index 2 is <start>, 1 is <unk> ('a'), others are mapped, 3 is <end>, 0 is <pad>
    
    decoded = tokenizer.decode(encoded, skip_special=True)
    print(f"Decoded back: '{decoded}'")
