# Ptv Hf Dataset With Token Encode
A PyTorch Dataset class wrapper for a Hugging Face dataset that converts text to token IDs.

    Args:  
        name (str): Name of the dataset class.  
        split (str): The name of the subset of dataset to retrieve or the split specifier.
        sample_field_name (str): Field name for text samples.
        label_field_name (str): Field name for labels.
        encode (PTCALLABLE): The reference to a token encoder function.
        remove_html_tags (bool): Remove html tags in text if True.
        encode_return_dict (bool): Encode function returns a Dict instead of a Tuple.

## Input
| Name | Data type |
|---|---|
| name | String |
| split | String |
| sample_field_name | String |
| label_field_name | String |
| encode | Ptcallable |
| remove_html_tags | Boolean |
| encode_return_dict | Boolean |

## Output
| Data type |
|---|
| Ptvdataset |

<HR>
Category: PyTorch wrapper - Training

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
