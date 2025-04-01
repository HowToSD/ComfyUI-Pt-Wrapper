from typing import List, Dict, Any, Tuple, Type
import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType
from peft.tuners.lora import Linear as LoRALinear

class HfLoraClassificationModel(nn.Module):
    """Generic transformer-based binary classification model with LoRA adaptation.

    This wrapper supports Hugging Face encoder models fine-tuned via LoRA. It supports 
    three pooling strategies for sequence classification: mean pooling, pooler output, 
    or CLS token embedding. Only LoRA-adapted attention layers and the classification 
    head are trainable; all base model parameters remain frozen.

    Args:
        model_name (str): Name or path of the pretrained Hugging Face model.
        use_mean_pooling (bool): If True, average token embeddings using the attention mask.
                                 Otherwise, use pooler output (if available) or the CLS token.
        dropout (float): Dropout rate applied before the final classification layer.
                         Set to 0.0 to disable dropout. Must be ≥ 0.
        lora_r (int): LoRA rank parameter (dimension of the low-rank matrices).
        lora_alpha (int): LoRA scaling factor. The adapted weight is added as:
                          (lora_alpha / lora_r) * (A @ B), where
                          A is a (m × r) matrix and B is a (r × n) matrix.
        lora_dropout (float): Dropout applied to the input of the LoRA layers.
    
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
                 dropout: float = 0.1,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1):
        super().__init__()
        if dropout < 0.0:
            raise ValueError(f"Dropout must be non-negative, got {dropout}")

        self.config = AutoConfig.from_pretrained(model_name)
        model_class = self._resolve_model_class(model_name)
        base_model = model_class.from_pretrained(model_name, config=self.config)

        # Apply LoRA
        target_modules = [
            # BERT, RoBERTa, ALBERT, Electra
            "query", "key", "value", "dense",

            # DistilBERT
            "q_lin", "k_lin", "v_lin", "out_lin",

            # GPT-2
            "c_attn", "c_proj",

            # LLaMA, Ollama, Visual LLaMA
            "q_proj", "k_proj", "v_proj", "o_proj",

            # CLIP (Text & Vision Transformers)
            "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj",
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
            "proj",  # common for CLIP-ViT, ViT heads

            # T5
            "q", "k", "v", "o",
            "wi_0", "wi_1", "wo",

            # ViT (Hugging Face ViTModel, etc.)
            "attention.query", "attention.key", "attention.value", "attention.output.dense",
            "mlp.dense_in", "mlp.dense_out"

            # FFN modules
            "ffn.lin1", "ffn.lin2",                      # DistilBERT
            "intermediate.dense", "output.dense",       # BERT, RoBERTa, Electra
            "wi_0", "wi_1", "wo",                        # T5
            "mlp.dense_in", "mlp.dense_out",            # ViT
            "mlp.c_fc", "mlp.c_proj",                   # GPT-2
            "mlp.up_proj", "mlp.down_proj", "mlp.gate_proj"  # LLaMA-family
        ]

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            # target_modules=target_modules,
            target_modules="all-linear",  # Inject to all linear layers
            lora_dropout=lora_dropout,
            bias="lora_only",  # Makes the bias in base model layers trainable only if those layers have LoRA applied.
            task_type=TaskType.FEATURE_EXTRACTION
        )
        self.llm_model = get_peft_model(base_model, peft_config)

        self.lora_injected_layers = []
        for name, module in self.llm_model.named_modules():
            if isinstance(module, LoRALinear):
                self.lora_injected_layers.append(name)

        if not self.lora_injected_layers:
            raise ValueError("LoRA injected into zero modules — check target_modules.")

        # for name in self.lora_injected_layers:
        #     print(f"[LoRA] Injected into: {name}")

        self.hidden_size = self.config.hidden_size
        self.use_mean_pooling = use_mean_pooling
        self.linear = nn.Linear(self.hidden_size, 1)

        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        # Freeze base model weights except LoRA and linear
        for param in self.llm_model.base_model.parameters():
            param.requires_grad = False
        for param in self.llm_model.parameters():
            if param.requires_grad:  # only LoRA params
                param.requires_grad = True
        for param in self.linear.parameters():
            param.requires_grad = True


    def _resolve_model_class(self, model_name: str) -> Type[PreTrainedModel]:
        for prefix, class_name in self.MODEL_CLASS_PREFIXES:
            if model_name.startswith(prefix):
                module = __import__("transformers", fromlist=[class_name])
                return getattr(module, class_name)
        raise ValueError(f"Model '{model_name}' is not supported.")

    def forward(self, inputs:Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Runs a forward pass through the transformer model followed by optional dropout and a classification head.

        Args:
            inputs (Tuple[torch.Tensor, torch.Tensor]): Tuple of below tensors:
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
