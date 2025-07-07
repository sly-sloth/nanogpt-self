from typing import List

# TODO:
# 1. Can add unknown token for foreign chars.
# 2. C++/Rust for faster implementation
# 3. Pre load a-z and A-Z
# 4. Trie implementation for single parse through text to encode


class BPETokenizer:
    def __init__(self):
        self.itos = {}
        self.stoi = {}
        self.vocab_size = 0
        self.max_key = 0
        self.merges = {}
    
    def train(self, text_corpus: str, vocab_size: int, verbose: bool = False):
        """
        Takes in the corpus of text to train on and the vocabulary size.
        
        Input: 
            text_corpus: str - The corpus of text to train the tokenizer on.

            vocab_size: int - The size of the vocabulary of the tokenizer.
        """
        self.vocab_size = vocab_size
        text_corpus_set = set(text_corpus)
        text_corpus_unique = sorted(text_corpus_set)

        assert len(text_corpus_unique) <= vocab_size, f"vocab_size ({vocab_size}) must be greater than or equal to the number of unique chars ({len(text_corpus_unique)}) in the text corpus"

        for ch in text_corpus_unique:
            self.stoi[ch] = self.max_key
            self.itos[self.max_key] = ch
            self.max_key += 1

        text_idx = [self.stoi[ch] for ch in text_corpus]

        milestones = [0] + [int(vocab_size * frac) - 1 for frac in (0.1 * i for i in range(2, 11, 2))]

        while self.max_key < vocab_size:
            if verbose:
                if self.max_key in milestones:
                    percent_trained = int(round((100 * (self.max_key + 1) / vocab_size)))
                    print(f"Training progress: {percent_trained}%")

            pair_count = {}
            for i1, i2 in zip(text_idx, text_idx[1:]):
                pair_count[(i1, i2)] = pair_count.get((i1, i2), 0) + 1

            max_pair = max(pair_count, key=lambda pair: pair_count[pair])
            new_text_idx = []

            i = 0
            while i < len(text_idx)-1:
                if (text_idx[i], text_idx[i+1]) == max_pair:
                    new_text_idx.append(self.max_key)
                    i += 2
                else:
                    new_text_idx.append(text_idx[i])
                    i += 1
            
            merged_pair = self.itos[max_pair[0]] + self.itos[max_pair[1]]
            self.itos[self.max_key] = merged_pair
            self.stoi[merged_pair] = self.max_key

            self.merges[max_pair] = self.max_key
            self.max_key += 1
            text_idx = new_text_idx

        self.final_vocab = sorted(self.stoi.items(), key=lambda item: len(item[0]), reverse=True)
        print("Tokenizer successfully trained!")
        return
    
    def encode(self, text: str):
        """
        Function to encode a given string.

        Input:
            text: str - The text (string) which is to be encoded.
        """
        encoded_idx = []
        rem_text = text

        while rem_text:
            for substr, subkey in self.final_vocab:
                if rem_text.startswith(substr):
                    encoded_idx.append(subkey)
                    rem_text = rem_text[len(substr):]
                    break
        
        return encoded_idx

    def decode(self, idx: List[int]):
        """
        Function to decode a given list of encoded indices.

        Input:
            idx: List[int] - The list of encoded indices (encoding) which is to be decoded.
        """
        return "".join(self.itos[i] for i in idx)
