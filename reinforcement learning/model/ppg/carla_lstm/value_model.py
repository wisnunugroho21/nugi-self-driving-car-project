import torch
import torch.nn as nn

class ValueModel(nn.Module):
    def __init__(self, state_dim):
      super(ValueModel, self).__init__()

      self.state_extractor      = nn.Sequential( nn.Linear(2, 32), nn.ReLU() )
      self.image_extractor      = nn.LSTM(128, 128)
      
      self.nn_layer             = nn.Sequential( nn.Linear(160, 64), nn.ReLU() )
      self.critic_layer         = nn.Sequential( nn.Linear(64, 1) )
        
    def forward(self, res, state, detach = False):
      out_i, _ = self.image_extractor(res)
      i = out_i[-1]

      s   = self.state_extractor(state)
      x   = torch.cat([i, s], -1)
      x   = self.nn_layer(x)

      if detach:
        return self.critic_layer(x).detach()
      else:
        return self.critic_layer(x)