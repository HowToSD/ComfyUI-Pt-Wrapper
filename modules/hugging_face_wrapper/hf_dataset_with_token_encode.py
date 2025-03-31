from typing import Tuple, Callable, Optional
import re
from html import unescape
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
                 encode: Optional[Callable]=None,
                 remove_html_tags:Optional[bool]=False,
                 encode_return_dict:Optional[bool]=False) -> None:
        """
        Initializes the dataset by loading Hugging Face dataset.

        Args:
            dataset_name (str): The name of the Hugging Face dataset.
            split (str): The dataset split (e.g., 'train').
             sample_field_name (str): Field name for text samples.
            label_field_name (str): Field name for labels.
            encode (Optional[Callable]): Token encode function
            remove_html_tags (Optional[bool]): Remove html tags in text if True.
            encode_return_dict (Optional[bool]): Encode function returns a Dict instead of a Tuple.
        """
        self.dataset = load_dataset(dataset_name, split=split)
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
            # Decode HTML entities like "&amp;", "&lt;", etc. to their actual characters
            text = unescape(sample[self.sample_field_name])
            
            # Remove HTML tags (like <br />, <p>, etc.)
            # The regex "<.*?>" uses .*? (non-greedy match) to ensure it matches one tag at a time,
            # instead of greedily removing everything between the first '<' and last '>'
            text = re.sub(r"<.*?>", " ", text)

            # Replace consecutive white space to a single white space
            text = re.sub(r"  +", " ", text)
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
