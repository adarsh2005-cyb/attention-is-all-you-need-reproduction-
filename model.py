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
