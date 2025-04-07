# Ptn Embedding
Constructs an embedding layer.

    Args:
        num_embeddings (int): Vocabulary size.  
        embedding_dim (int): The size of each embedding vector.  
        padding_idx (int): Token ID to treat as padding. If a non-negative integer is specified, the embedding at that index will be excluded from gradient updates. Set to -1 to disable padding behavior (default).

## Input
| Name | Data type |
|---|---|
| num_embeddings | Int |
| embedding_dim | Int |
| padding_idx | Int |

## Output
| Data type |
|---|
| Ptmodel |

<HR>
Category: PyTorch wrapper - Model

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
