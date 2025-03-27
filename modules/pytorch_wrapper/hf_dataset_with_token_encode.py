from typing import Tuple, Callable
import torch
from torch.utils.data import Dataset
from datasets import load_dataset  # Hugging Face (not PyTorch)


class HfDatasetWithTokenEncode(Dataset):
    """
    Hugging Face Dataset wrapper with custom token encode function.

    pragma: skip_doc
    """
    def __init__(self,
                 dataset_name: str,
                 split: str,
                 sample_field_name: str = "text",
                 label_field_name: str = "label",
                 encode: Callable=None) -> None:
        """
        Initializes the dataset by loading Hugging Face dataset.

        Args:
            dataset_name (str): The name of the Hugging Face dataset.
            split (str): The dataset split (e.g., 'train').
             sample_field_name (str): Field name for text samples.
            label_field_name (str): Field name for labels.
            encode (Callable): Token encode function
        """
        self.dataset = load_dataset(dataset_name, split=split)
        self.sample_field_name = sample_field_name
        self.label_field_name = label_field_name
        self.encode = encode


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the (tokens, label) pair at the given index.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: Tuple of the following:  
                - Tokens tensor and masks tensor.
                - Label tensor
        """
        # Get a text sample from database and convert to a word embedding
        sample = self.dataset[idx]
        tokens, masks = self.encode(sample[self.sample_field_name])

        # Get a corresponding label
        label = torch.tensor(sample[self.label_field_name], dtype=torch.long)
        return (tokens, masks), label

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Dataset length.
        """
        return len(self.dataset)
