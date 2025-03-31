from typing import List, Dict, Any, Tuple, Type
import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel

class HfFineTunedClassificationModel(nn.Module):
    """Generic transformer-based binary classification model.

    Args:
        model_name (str): Name or path of the pretrained Hugging Face model.
        use_mean_pooling (bool): If True, average token embeddings using the attention mask.
                                 Otherwise, use pooler output or CLS token embedding.
        dropout (float): Dropout rate applied before the final linear layer.
                         Set to 0.0 to disable dropout. Must be ≥ 0.

    pragma: skip_doc
    """

    MODEL_CLASS_PREFIXES: List[Tuple[str, str]] = [
        ("bert-base-uncased", "BertModel"),
        ("roberta-base", "RobertaModel"),
        ("distilbert-base-uncased", "DistilBertModel"),
        ("albert-base-v2", "AlbertModel"),
        ("google/electra-base-discriminator", "ElectraModel"),
    ]

    def __init__(self,
                 model_name: str,
                 use_mean_pooling: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        if dropout < 0.0:
            raise ValueError(f"Dropout must be non-negative, got {dropout}")

        self.config = AutoConfig.from_pretrained(model_name)
        model_class = self._resolve_model_class(model_name)
        self.llm_model: PreTrainedModel = model_class.from_pretrained(model_name, config=self.config)

        self.hidden_size = self.config.hidden_size
        self.use_mean_pooling = use_mean_pooling
        self.linear = nn.Linear(self.hidden_size, 1)

        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

        # Initialize linear weight
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def _resolve_model_class(self, model_name: str) -> Type[PreTrainedModel]:
        for prefix, class_name in self.MODEL_CLASS_PREFIXES:
            if model_name.startswith(prefix):
                module = __import__("transformers", fromlist=[class_name])
                return getattr(module, class_name)
        raise ValueError(f"Model '{model_name}' is not supported.")

    def forward(self, inputs:Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Runs a forward pass through the transformer model followed by optional dropout and a classification head.

        Args:
            inputs (Tuple[torch.Tensor]): Tuple of below tensors:
                input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_len).
                attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Raw classification logits of shape (batch_size, 1).
        """
        input_ids, attention_mask = inputs
        outputs = self.llm_model(input_ids=input_ids, attention_mask=attention_mask)

        if self.use_mean_pooling:
            last_hidden = outputs.last_hidden_state
            # Expand the mask across the hidden dimension
            # e.g. if hidden size is 768, the (seq_len,) mask becomes (seq_len, 768) per sample.
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
            mask_adjusted_count = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            x = sum_embeddings / mask_adjusted_count
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            x = outputs.pooler_output
        else:  # Grab the first feature in the sequence
            x = outputs.last_hidden_state[:, 0, :]

        assert x.dim() == 2

        if self.use_dropout:
            x = self.dropout(x)
        x = self.linear(x)
        return x
