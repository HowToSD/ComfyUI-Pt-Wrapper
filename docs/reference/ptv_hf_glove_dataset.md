# Ptv Hf Glove Dataset
A PyTorch Dataset class wrapper for a Hugging Face dataset that converts text to embedding using Glove.

    Args:  
        name (str): Name of the dataset class (e.g., MNIST, FashionMNIST).  
        split (str): The name of the subset of dataset to retrieve or the split specifier.
        embed_dim (int): Embedding dimension for Glove.
        max_seq_len (int): Sequence length.
        sample_field_name (str): Field name for text samples.
        label_field_name (str): Field name for labels.

## Input
| Name | Data type |
|---|---|
| name | String |
| split | String |
| embed_dim | Int |
| max_seq_len | Int |
| sample_field_name | String |
| label_field_name | String |

## Output
| Data type |
|---|
| Ptvdataset |

<HR>
Category: PyTorch wrapper - Training

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
