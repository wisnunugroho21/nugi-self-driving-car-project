import torch
import torch.nn as nn

from helpers.pytorch_utils import set_device

class Cpc(nn.Module):
    def __init__(self, use_gpu = True):
        super(Cpc, self).__init__()

        self.W = nn.Parameter(torch.rand(128, 128)).to(set_device(use_gpu))

    def compute_logits(self, encoded_anchor, encoded_target):
        Wz      = torch.matmul(self.W, encoded_target.T)  # (z_dim,B)
        logits  = torch.matmul(encoded_anchor, Wz)  # (B,B)
        logits  = logits - torch.max(logits, 1)[0][:, None]

        return logits