import os
import sys
from typing import Tuple, Callable, Optional
import re
from html import unescape
import torch
from torch.utils.data import Dataset
from datasets import load_dataset  # Hugging Face (not PyTorch)
from .utils import drop_html_tags

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.utils import get_dataset_full_path

class HfLocalDataset(Dataset):
    """
    Hugging Face Dataset wrapper to load datasets stored on the local file system.

    pragma: skip_doc
    """
    def __init__(self,
                 file_path: str,
                 file_format: str,
                 sample_field_name: str = "text",
                 label_field_name: str = "label",
                 encode: Optional[Callable]=None,
                 remove_html_tags:Optional[bool]=False,
                 encode_return_dict:Optional[bool]=False) -> None:
        """
        Initializes the dataset by loading Hugging Face dataset.

        Args:
            file_path (str): Path to the dataset file. Can be an absolute path or a relative path.  
            If a relative path is provided, it is resolved from the `datasets` directory under this extension.  
            For example, to load `datasets/foo/bar_train.jsonp`, specify `foo/bar_train.jsonp` or the absolute path.
            file_format (str): File format (e.g. json if the file is in jsonl format).
            sample_field_name (str): Field name for text samples.
            label_field_name (str): Field name for labels.
            encode (Optional[Callable]): Token encode function
            remove_html_tags (Optional[bool]): Remove html tags in text if True.
            encode_return_dict (Optional[bool]): Encode function returns a Dict instead of a Tuple.
        """
        dataset_full_path = get_dataset_full_path(file_path)

        # Note split="train" is used here to load the whole dataset
        # and not to load just the train dataset.
        self.dataset = load_dataset(file_format,
                                    data_files=dataset_full_path,
                                    split="train")
        self.sample_field_name = sample_field_name
        self.label_field_name = label_field_name
        self.encode = encode
        self.remove_html_tags = remove_html_tags
        self.encode_return_dict = encode_return_dict

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
        if self.remove_html_tags:
            text = drop_html_tags(sample[self.sample_field_name])
        else:
            text = sample[self.sample_field_name]

        if self.encode_return_dict:
            output = self.encode(text)
            tokens = output["input_ids"]
            masks = output["attention_mask"]
        else:
            tokens, masks = self.encode(text)

        # Get a corresponding label
        label = torch.tensor(sample[self.label_field_name], dtype=torch.long)
        return (tokens.squeeze(0), masks.squeeze(0)), label.squeeze(0)


    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Dataset length.
        """
        return len(self.dataset)
