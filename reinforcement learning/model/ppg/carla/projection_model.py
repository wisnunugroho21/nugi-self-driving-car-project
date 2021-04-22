import torch.nn as nn
import torch

from helpers.pytorch_utils import set_device

class ProjectionModel(nn.Module):
    def __init__(self):
      super(ProjectionModel, self).__init__()

      self.nn_layer   = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
      )

      self.W = nn.Parameter(torch.rand(32, 32))

    def forward(self, res, detach = False):      
      if detach:
        return self.nn_layer(res).detach()
      else:
        return self.nn_layer(res)

    def compute_logits(self, encoded_anchor, encoded_target):
      Wz      = torch.matmul(self.W, encoded_target.T)  # (z_dim,B)
      logits  = torch.matmul(encoded_anchor, Wz)  # (B,B)
      logits  = logits - torch.max(logits, 1)[0][:, None]

      return logits