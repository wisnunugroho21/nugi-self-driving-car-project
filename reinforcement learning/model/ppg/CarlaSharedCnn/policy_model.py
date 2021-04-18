import torch
import torch.nn as nn

from helpers.pytorch_utils import set_device

class PolicyModel(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(PolicyModel, self).__init__()

      self.std                  = torch.FloatTensor([1.0, 0.5, 0.5]).to(set_device(use_gpu))

      self.state_extractor      = nn.Sequential( nn.Linear(2, 32), nn.ReLU() )
      self.image_extractor      = nn.Sequential( nn.Linear(128, 128), nn.ReLU() )
      self.nn_layer             = nn.Sequential( nn.Linear(160, 128), nn.ReLU() )
      
      self.actor_steer          = nn.Sequential( nn.Linear(128, 1), nn.Tanh() )
      self.actor_gas_break      = nn.Sequential( nn.Linear(128, 2), nn.Sigmoid() )

      self.critic_layer         = nn.Sequential( nn.Linear(128, 1) )
        
    def forward(self, res, state, detach = False):
      i   = self.image_extractor(res)
      s   = self.state_extractor(state)
      x   = torch.cat([i, s], -1)
      x   = self.nn_layer(x)

      action_steer        = self.actor_steer(x)
      action_gas_break    = self.actor_gas_break(x)

      action              = torch.cat((action_steer, action_gas_break), -1)
      critic              = self.critic_layer(x)

      if detach:
        return (action.detach(), self.std.detach()), critic.detach()
      else:
        return (action, self.std), critic