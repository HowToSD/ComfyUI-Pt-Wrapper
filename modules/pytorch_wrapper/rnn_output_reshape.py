# Return Value of RNN Model's forward()
"""
model = nn.RNN(...)  # or nn.LSTM, nn.GRU
out_seq, _ = model(input)

out_seq has shape:
    (batch, seq_len, hidden_size)  if batch_first=True

To extract the output from the last time step of the top RNN layer:

Unidirectional Case:
    final_output = out_seq[:, -1, :]  # shape: (batch, hidden_size)

Bidirectional Case:
    If the model is bidirectional, hidden_size in out_seq is already doubled.
    No need to manually concatenate forward/backward states.
    
    final_output = out_seq[:, -1, :]  # shape: (batch, 2 * hidden_size)

To match label shape:
    - If the label is 3D: shape (batch, seq_len, hidden_size), use label[:, -1, :]
    - If bidirectional: repeat the last axis
"""
from typing import Tuple, Union
import torch

def extract_rnn_return_value_with_adjusted_label(
    inputs: torch.Tensor,
    label: torch.Tensor,
    rnn_output: torch.Tensor,
    bidirectional: bool,
    batch_first:bool = True,
    process_label:bool=True,
    return_valid_token_mean:bool=True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extracts the final output at the last time step from the top RNN layer (out_seq[:, -1, :])
    and adjusts the label shape to match it.

    Args:
        inputs (torch.Tensor): The input (feature) tensor.
        label (torch.Tensor): Ground truth tensor of shape (batch, seq_len, hidden_size).
        rnn_output (torch.Tensor): The output sequence from the RNN (out_seq), 
            shape (batch, seq_len, hidden_size) for unidirectional,
            or (batch, seq_len, 2 * hidden_size) for bidirectional RNNs.
        bidirectional (bool): Whether the RNN is bidirectional.
        batch_first (bool): True if the first axis is the batch.
        process_label (bool): True if process label. To process only the RNN output, pass None to label and set this flag to False.
        return_valid_token_mean (bool): True to return the mean of non-zero outputs.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The final time-step output of shape (batch, hidden_size) or (batch, 2 * hidden_size)
            - The adjusted label of matching shape
    """
    if batch_first:
        if return_valid_token_mean:
            # Use inputs to build a valid token mask
            valid_mask = inputs.norm(p=2, dim=2) > 0  # shape: [batch, seq_len]
            lengths = valid_mask.sum(dim=1).clamp(min=1)       # shape: [batch]
            masked = rnn_output * valid_mask.unsqueeze(-1)     # [batch, seq_len, hidden]
            out = masked.sum(dim=1) / lengths.unsqueeze(1)     # [batch, hidden]
        else:
            out = rnn_output[:, -1, :]  # last time step output from top layer
        if process_label:
            label = label[:, -1, :]
    else:
        if return_valid_token_mean:
            valid_mask = inputs.norm(p=2, dim=2) > 0  # shape: [seq_len, batch]
            lengths = valid_mask.sum(dim=0).clamp(min=1)       # shape: [batch]
            masked = rnn_output * valid_mask.unsqueeze(-1)     # [seq_len, batch, hidden]
            out = masked.sum(dim=0) / lengths.unsqueeze(1)     # [batch, hidden]
        else:
            out = rnn_output[-1, :, :]  # last time step output from top layer
        if process_label:
            label = label[-1, :, :]

    if bidirectional and process_label:
        label = torch.cat([label, label], dim=-1)

    if process_label:
        return out, label
    else:
        return out
