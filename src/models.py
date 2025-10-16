"""
models.py

This script defines the neural network architectures used for the T2D prediction task.
It includes two models:
1.  A standard LSTM model to serve as a performance baseline.
2.  The RETAIN model, an interpretable architecture based on Choi et al., 2016.

Both models are implemented using PyTorch.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """
    A standard LSTM baseline model for sequence classification.

    This model consists of an embedding layer to convert integer-coded medical
    concepts into dense vectors, an LSTM layer to process the sequence of
    visits, and a final fully-connected layer with a sigmoid activation to
    produce a binary prediction.

    Args:
        vocab_size (int): The total number of unique medical codes in the dataset.
        embedding_dim (int): The dimensionality of the embedding vectors.
        hidden_size (int): The number of features in the LSTM hidden state.
        output_size (int): The number of output features (1 for binary classification).
        num_layers (int): The number of recurrent layers in the LSTM.
        dropout (float): The dropout probability for the LSTM layer.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, output_size: int, num_layers: int = 1, dropout: float = 0.5):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the LSTM model.

        Args:
            x (torch.Tensor): A tensor of patient sequences.
                              Shape: (batch_size, seq_length).

        Returns:
            torch.Tensor: The model's prediction. Shape: (batch_size, 1).
        """
        # Input shape: (batch_size, seq_length)
        embedded = self.embedding(x)
        # -> Shape: (batch_size, seq_length, embedding_dim)

        lstm_out, _ = self.lstm(embedded)
        # -> Shape: (batch_size, seq_length, hidden_size)

        # We use the hidden state of the final time step for classification
        final_hidden_state = lstm_out[:, -1, :]
        # -> Shape: (batch_size, hidden_size)

        out = self.fc(final_hidden_state)
        # -> Shape: (batch_size, output_size)

        out = self.sigmoid(out)
        return out


class RETAIN(nn.Module):
    """
    The RETAIN model for interpretable predictions (Choi et al., 2016).

    This model uses a two-level attention mechanism to generate predictions.
    It calculates visit-level attention (alpha) and code-level attention (beta)
    to create a weighted context vector, which is then used for the final
    classification. This structure allows for inspection of the attention
    weights to interpret model decisions.

    Args:
        vocab_size (int): The total number of unique medical codes.
        embedding_dim (int): The dimensionality of the embedding vectors.
        hidden_size (int): The number of features in the hidden state of the GRUs.
        output_size (int): The number of output features (1 for binary classification).
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, output_size: int):
        super(RETAIN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # RNN to generate visit-level attention weights (alpha)
        self.rnn_alpha = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.alpha_fc = nn.Linear(hidden_size, 1)

        # RNN to generate medical code-level attention weights (beta)
        self.rnn_beta = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.beta_fc = nn.Linear(hidden_size, embedding_dim)

        # Final output layer
        self.output_fc = nn.Linear(embedding_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """
        Defines the forward pass for the RETAIN model.

        Args:
            x (torch.Tensor): A tensor of patient sequences.
                              Shape: (batch_size, seq_length).
            return_attention (bool): If True, returns the attention weights
                                     along with the prediction.

        Returns:
            torch.Tensor | Tuple[torch.Tensor, ...]:
                - If return_attention is False: The model's prediction.
                  Shape: (batch_size, 1).
                - If return_attention is True: A tuple containing the prediction,
                  alpha weights, and beta weights.
        """
        # Input shape: (batch_size, seq_length)
        embedded = self.embedding(x)
        # -> Shape: (batch_size, seq_length, embedding_dim)

        # --- Alpha (visit-level attention) ---
        alpha_out, _ = self.rnn_alpha(embedded)
        # -> Shape: (batch_size, seq_length, hidden_size)
        alpha = self.alpha_fc(alpha_out).squeeze(-1)
        # -> Shape: (batch_size, seq_length)
        alpha_weights = F.softmax(alpha, dim=1)
        # -> Shape: (batch_size, seq_length)

        # --- Beta (code-level attention) ---
        beta_out, _ = self.rnn_beta(embedded)
        # -> Shape: (batch_size, seq_length, hidden_size)
        beta_weights = torch.tanh(self.beta_fc(beta_out))
        # -> Shape: (batch_size, seq_length, embedding_dim)

        # --- Context vector construction ---
        # Weight the embeddings by both alpha and beta attention scores
        context_vector = torch.sum(alpha_weights.unsqueeze(2) * beta_weights * embedded, dim=1)
        # -> Shape: (batch_size, embedding_dim)

        # --- Final prediction ---
        output = self.output_fc(context_vector)
        # -> Shape: (batch_size, output_size)
        output = self.sigmoid(output)

        if return_attention:
            return output, alpha_weights, beta_weights
        else:
            return output