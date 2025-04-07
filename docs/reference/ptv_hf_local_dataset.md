# Ptv Hf Local Dataset
A PyTorch Dataset class wrapper for a Hugging Face dataset to load a dataset that is stored on the local file system.

    Args:  
        file_path (str): Path to the dataset file. Can be an absolute path or a relative path.  
        If a relative path is provided, it is resolved from the `datasets` directory under this extension.  
        For example, to load `datasets/foo/bar_train.jsonp`, specify `foo/bar_train.jsonp` or the absolute path.
        file_format (str): File format (e.g. json if the file is in jsonl format).  
        sample_field_name (str): Field name for text samples.
        label_field_name (str): Field name for labels.
        encode (PTCALLABLE): The reference to a token encoder function.
        remove_html_tags (bool): Remove html tags in text if True.
        encode_return_dict (bool): Encode function returns a Dict instead of a Tuple.

## Input
| Name | Data type |
|---|---|
| file_path | String |
| file_format | String |
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
