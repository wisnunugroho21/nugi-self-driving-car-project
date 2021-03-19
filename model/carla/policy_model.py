import torch
import torch.nn as nn

from helper.pytorch import set_device

class PolicyModel(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(PolicyModel, self).__init__()

      self.std                  = torch.FloatTensor([1.0, 0.5, 0.5]).to(set_device(use_gpu))

      self.state_extractor      = nn.Sequential( nn.Linear(1, 32), nn.ReLU() )
      self.nn_layer             = nn.Sequential( nn.Linear(160, 320), nn.ReLU(), nn.Linear(320, 128), nn.ReLU() )

      self.critic_layer         = nn.Sequential( nn.Linear(128, 1) )
      self.actor_tanh_layer     = nn.Sequential( nn.Linear(128, 1), nn.Tanh() )
      self.actor_sigmoid_layer  = nn.Sequential( nn.Linear(128, 2), nn.Sigmoid() )            
        
    def forward(self, res, state, detach = False):
      s   = self.state_extractor(state)
      x   = torch.cat([res, s], -1)
      x   = self.nn_layer(x)

      action_tanh     = self.actor_tanh_layer(x)
      action_sigmoid  = self.actor_sigmoid_layer(x)
      action          = torch.cat((action_tanh, action_sigmoid), -1)

      if detach:
        return (action.detach(), self.std.detach()), self.critic_layer(x).detach()
      else:
        return (action, self.std), self.critic_layer(x)