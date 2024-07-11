import torch
from torch import nn, Tensor
import math

import pytorch_lightning as pl

class PositionalEncoding(pl.LightningModule):
    def __init__(self, d_model, max_len=5000, base=576, device=None):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).to(device)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).requires_grad_(False)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]
    
if __name__ == "__main__":
    pe = PositionalEncoding(4096)
    x = 3