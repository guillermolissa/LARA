import torch
import torch.nn as nn


class LSTMModel(nn.Module):

  def __init__(self, embedded_dim, hidden_dim, layer_dim, items_size, drop_rate:float=0):
    super(LSTMModel, self).__init__()

    self.hidden_dim = hidden_dim
    self.layer_dim = layer_dim


    self.embedding = nn.Embedding(items_size, embedded_dim)

    self.lstm = nn.LSTM(embedded_dim, hidden_dim, num_layers=layer_dim, dropout=drop_rate, batch_first=True)
    self.fc = nn.Linear(hidden_dim, items_size)

  def forward(self, x, h0=None, c0=None):

    # Initialize hidden state and cell state if not provided
    if h0 is None or c0 is None:
      h0 = torch.zeros(self.layer_dim, x.size(
          0), self.hidden_dim).to(x.device)
      c0 = torch.zeros(self.layer_dim, x.size(
          0), self.hidden_dim).to(x.device)


    embedded = self.embedding(x)

    """
    out (intermediate_hidden_states): Tensor of shape (batch_size, seq_length, hidden_size) containing the output features (h_t) from the last layer of the LSTM, for each t.
    hn (final_hidden_state): Tensor of shape (num_layers * num_directions, batch_size, hidden_size) containing the final hidden state for each element in the sequence.
    cn (final_cell_state): Tensor of shape (num_layers * num_directions, batch_size, hidden_size) containing the final cell state for each element in the sequence.
    
    NOTE: out will contain the hidden states for all time steps, while hn and cn will contain the hidden and cell states for the last time step only.
          it will be used as input to attention mechanism, while hn and cn will be used to initialize the hidden and cell states for the next batch of sequences.

    """


    out, (hn, cn) = self.lstm(embedded, (h0, c0))
    output = self.fc(hn.squeeze(0))
    return output, hn, cn



class LSTMAttentionModel(nn.Module):

  def __init__(self, embedded_dim, hidden_dim, layer_dim, items_size, n_head = 4, drop_rate = 0.1, context_length = 10):

    super(LSTMAttentionModel, self).__init__()

    self.hidden_dim = hidden_dim
    self.layer_dim = layer_dim
    self.kdim = hidden_dim
    self.vdim = hidden_dim


    self.embedding = nn.Embedding(items_size, embedded_dim)

    if layer_dim > 1:
        self.lstm = nn.LSTM(embedded_dim, hidden_dim, num_layers=layer_dim, batch_first=True, dropout=drop_rate)
    else:
        self.lstm = nn.LSTM(embedded_dim, hidden_dim, num_layers=layer_dim, batch_first=True)

    # multihead attention
    self.q = nn.Linear(hidden_dim, self.kdim)
    self.k = nn.Linear(hidden_dim, self.kdim)
    self.v = nn.Linear(hidden_dim, self.vdim)

    # Correct: Registering the tensor as a buffer. This ensures that the tensor is moved to the appropriate device when the model is moved to a different device (e.g., GPU).
    self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )


    self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_head, dropout=drop_rate,  bias=False, kdim = self.kdim, vdim = self.vdim, batch_first = True)

    self.norm = nn.LayerNorm(hidden_dim)

    self.fc = nn.Linear(hidden_dim, items_size)

  def attention(self, lstm_hidden_layer):
    # convert to query, key and the value matrixes. 
    query = self.q(lstm_hidden_layer)
    key = self.k(lstm_hidden_layer)
    value = self.v(lstm_hidden_layer)
    # can input batch: because output is (N,L,E) when batch_first=True
    attention_out, weights = self.multihead_attention(query, key, value, attn_mask=self.mask)
    return attention_out
    

  def forward(self, x, h0=None, c0=None):

    # Initialize hidden state and cell state if not provided
    if h0 is None or c0 is None:
      h0 = torch.zeros(self.layer_dim, x.size(
          0), self.hidden_dim).to(x.device)
      c0 = torch.zeros(self.layer_dim, x.size(
          0), self.hidden_dim).to(x.device)


    embedded = self.embedding(x)

    lstm_out, (hn, cn) = self.lstm(embedded, (h0, c0))
    attended_out = self.attention(lstm_out)
    norm_out = self.norm(attended_out + lstm_out)
    output = self.fc(norm_out[:, -1, :])
    return output, hn, cn