import torch
import torch.nn as nn

from helper.pytorch import set_device

class QModel(nn.Module):
    def __init__(self, state_dim, action_dim):
      super(QModel, self).__init__()

      self.state_extractor      = nn.Sequential( nn.Linear(1, 32), nn.ReLU() )
      self.image_extractor      = nn.Sequential( nn.Linear(128, 128), nn.ReLU() )
      self.nn_layer             = nn.Sequential( nn.Linear(160, 160), nn.ReLU(), nn.Linear(160, 128), nn.ReLU() )

      self.critic_layer         = nn.Sequential( nn.Linear(32, 1) )
        
    def forward(self, res, state, action, detach = False):
      i   = self.image_extractor(res)
      s   = torch.cat([state, action], -1)
      s   = self.state_extractor(state)
      x   = torch.cat([i, s], -1)
      x   = self.nn_layer(x)

      if detach:
        return self.critic_layer(x).detach()
      else:
        return self.critic_layer(x)