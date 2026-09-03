#model mapping 
# 1- positional encoding
# 2- multi-head attention 
# 3- positional feed forward
# 4- encoder - decoder architecture 
# 5- burgering up the transformer 



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class PositionalEncoding(nn.Module):
  def __init__(self,d_model: int , max_len: int = 5000, dropout: float = 0.1,):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, d_model, 2, dtype=torch.float32).unsqueeze(1)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + pe[:, : x.size(1)]
    return self.dropout(x)
class MultiAttentionHead(nn.module):
  def __init__(self, d_model: int, n_heads: int, dropout: float= 0.1)
    super().__init__()
    assert d_model % n_heads == 0
    self.d_k = d_model/n_heads
    self.n_heads = n_heads
    self.w_q = nn.Linear(d_model , d_model)
    self.w_k = nn.Linear(d_model , d_model)
    self.w_v = nn.Linear(d_model , d_model)
    self.w_o = nn.Linear(d_model , d_model)
    self.dropout = nn.Dropout(dropout)

  def _split_heads(self, torch.Tensor) -> torch.Tensor:
    B ,L ,_ = x.shape
    return x.view(B , L , self.n_heads , self.d_k).transpose(1,2)

  def forward(self, query, key, value, mask=None):
    B = self.query.size(0)
    Q = self._split_heads(self.w_q(query))
    K = self._split_heads(self.k_q(key))
    V = self._split_heads(self.w_q(value))


    
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    attn = self.dropout(attn)
    out = torch.matmul(attn, V)  # (B, h, L, d_k)
    out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
    return self.w_o(out)
