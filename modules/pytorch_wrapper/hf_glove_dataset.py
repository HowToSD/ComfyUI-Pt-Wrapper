from typing import Tuple
import re
from html import unescape
import torch
from torch.utils.data import Dataset
from datasets import load_dataset  # Hugging Face (not PyTorch)
import scipy
import numpy as np
# Monkey-patch gensim as it relies on scipy's deprecated triu method
# /gensim/matutils.py has this line which causes ImportError:
# from scipy.linalg import get_blas_funcs, triu
# TODO: Monitor gensim and fix this if upgrade is made
scipy.linalg.triu = np.triu
import gensim.downloader as api

class HfGloveDataset(Dataset):
    """
    Hugging Face Dataset wrapper with word embedding support.

    pragma: skip_doc
    """
    def __init__(self,
                 dataset_name: str,
                 split: str,
                 glove_dim: int = 100,
                 max_seq_len: int = 256,
                 sample_field_name: str = "text",
                 label_field_name: str = "label") -> None:
        """
        Initializes the dataset by loading Hugging Face dataset and GloVe vectors.

        Args:
            dataset_name (str): The name of the Hugging Face dataset.
            split (str): The dataset split (e.g., 'train').
            glove_dim (int): Dimensionality of GloVe embeddings.
            max_seq_len (int): Maximum sequence length.
            sample_field_name (str): Field name for text samples.
            label_field_name (str): Field name for labels.
        """
        self.dataset = load_dataset(dataset_name, split=split)
        self.glove_dim = glove_dim
        self.max_seq_len = max_seq_len
        self.sample_field_name = sample_field_name
        self.label_field_name = label_field_name

        # Load GloVe from gensim
        glove_name = f"glove-wiki-gigaword-{glove_dim}"
        self.glove = api.load(glove_name)
        self.unk_vector = torch.zeros(glove_dim)

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenizes raw text into a list of lowercase word tokens.
        It removes HTML tags, decodes HTML entities, strips punctuation 
        (while keeping letters and numbers), and splits on whitespace.

        Args:
            text (str): Raw text input (may include HTML and punctuation).

        Returns:
            list[str]: List of clean, lowercase word tokens.
        """
        # Decode HTML entities like "&amp;", "&lt;", etc. to their actual characters
        text = unescape(text)
        
        # Remove HTML tags (like <br />, <p>, etc.)
        # The regex "<.*?>" uses .*? (non-greedy match) to ensure it matches one tag at a time,
        # instead of greedily removing everything between the first '<' and last '>'
        text = re.sub(r"<.*?>", " ", text)
        
        # Remove all characters except letters, digits, and whitespace
        # This strips punctuation like "!", ".", "(", but keeps words and numbers
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

        # Convert to lowercase and split by whitespace to get token list
        return text.lower().split()


    def vectorize(self, tokens: list[str]) -> torch.Tensor:
        """
        Converts tokens to a fixed-length tensor of word vectors.

        Args:
            tokens (list[str]): A list of word tokens.

        Returns:
            torch.Tensor: A tensor of shape (max_seq_len, glove_dim).
        """
        vectors = []
        for token in tokens[:self.max_seq_len]:
            if token in self.glove:
                vec = torch.tensor(self.glove[token], dtype=torch.float32)
            else:
                vec = self.unk_vector
            vectors.append(vec)

        if len(vectors) < self.max_seq_len:
            pad = [torch.zeros(self.glove_dim) for _ in range(self.max_seq_len - len(vectors))]
            vectors.extend(pad)

        return torch.stack(vectors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the (embedded sequence, label) pair at the given index.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Embedding tensor and label tensor.
        """
        # Get a text sample from database and convert to a word embedding
        sample = self.dataset[idx]
        tokens = self.tokenize(sample[self.sample_field_name])
        emb_tensor = self.vectorize(tokens)

        # Get a corresponding label
        label = torch.tensor(sample[self.label_field_name], dtype=torch.long)
        return emb_tensor, label

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Dataset length.
        """
        return len(self.dataset)
