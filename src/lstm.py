import torch
import torch.nn as nn
import json


def _build_meta_embedding(
    cfg: dict,
    tokenizer,
    emb_path: str,
    id_map_path: str,
) -> tuple[nn.Embedding, torch.Tensor]:
    """
    Load pretrained item metadata embeddings and build a bridge tensor that
    maps tokenizer indices → metadata embedding rows.

    The metadata embedding has an extra row 0 (zero vector, padding_idx) for
    tokenizer entries that have no corresponding metadata (special tokens,
    unknown items). All real item rows are shifted by +1.

    Returns:
        meta_emb    nn.Embedding [num_items+1, emb_dim]
        tok_to_meta LongTensor   [vocab_size]  — registered as a buffer
    """
    pretrained = torch.load(emb_path, map_location="cpu", weights_only=True)
    num_items, emb_dim = pretrained.shape

    assert emb_dim == cfg["emb_dim"], (
        f"Metadata embedding dim ({emb_dim}) != model emb_dim ({cfg['emb_dim']}). "
        f"Re-run build_item_embeddings.py with --embed_dim {cfg['emb_dim']}."
    )

    # Row 0 = zero padding; rows 1..N = pretrained embeddings (1-based shift)
    weight = torch.cat([torch.zeros(1, emb_dim), pretrained], dim=0)
    meta_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
    meta_emb.weight = nn.Parameter(weight)

    # id_map: {"item_id_str": metadata_row_index (0-based), ...}
    with open(id_map_path) as f:
        id_map: dict[str, int] = json.load(f)

    vocab = tokenizer.get_vocab()   # {"token_str": tokenizer_index, ...}
    vocab_size = tokenizer.get_vocab_size()

    # Build bridge: tokenizer_index → metadata_row (1-based), 0 = no metadata
    tok_to_meta = torch.zeros(vocab_size, dtype=torch.long)
    for token_str, tok_idx in vocab.items():
        if token_str in id_map:
            tok_to_meta[tok_idx] = id_map[token_str] + 1   # +1 for the padding row offset

    return meta_emb, tok_to_meta


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



class LSTMAttentionMetaEmbModel(nn.Module):

  def __init__(self, cfg, tokenizer=None):

    super(LSTMAttentionModel, self).__init__()

    self.drop_rate = cfg["drop_rate"] 
    self.hidden_dim = cfg["hidden_dim"]
    self.layer_dim = cfg["n_layers"]
    self.embedded_dim = cfg["emb_dim"]
    self.items_size = cfg["items_size"]
    self.contexlen = cfg["context_length"]
    self.kdim = cfg["hidden_dim"]
    self.vdim = cfg["hidden_dim"]
    self.n_heads = cfg["n_heads"]
    


    self.embedding = nn.Embedding(self.items_size,self.embedded_dim)

    self.use_meta = cfg["use_meta_embeddings"]
    if self.use_meta:
        assert tokenizer is not None, "tokenizer required when use_meta_embeddings=true"
        meta_emb, tok_to_meta = _build_meta_embedding(
            cfg, tokenizer,
            cfg["item_meta_embedding"],
            cfg["item_meta_id_map"],
        )
        self.meta_emb = meta_emb
        if cfg.get("freeze_meta_embeddings", True):
            self.meta_emb.weight.requires_grad_(False)
        self.register_buffer("tok_to_meta", tok_to_meta)


    if self.layer_dim > 1:
        self.lstm = nn.LSTM(self.embedded_dim, self.hidden_dim, num_layers=self.layer_dim, batch_first=True, dropout=self.drop_rate)
    else:
        self.lstm = nn.LSTM(self.embedded_dim, self.hidden_dim, num_layers=self.layer_dim, batch_first=True)

    # multihead attention
    self.q = nn.Linear(self.hidden_dim, self.kdim)
    self.k = nn.Linear(self.hidden_dim, self.kdim)
    self.v = nn.Linear(self.hidden_dim, self.vdim)

    # Correct: Registering the tensor as a buffer. This ensures that the tensor is moved to the appropriate device when the model is moved to a different device (e.g., GPU).
    self.register_buffer(
            "mask",
            torch.triu(torch.ones(self.contexlen , self.contexlen ),
                       diagonal=1)
        )


    self.multihead_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.n_heads, dropout=self.drop_rate,  bias=False, kdim = self.kdim, vdim = self.vdim, batch_first = True)

    self.norm = nn.LayerNorm(self.hidden_dim)

    self.fc = nn.Linear(self.hidden_dim, self.items_size)

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