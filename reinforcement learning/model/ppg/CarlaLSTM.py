import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers.pytorch_utils import set_device
from model.components.SeperableConv2d import DepthwiseSeparableConv2d

class CnnModel(nn.Module):
    def __init__(self):
      super(CnnModel, self).__init__()   

      self.conv1 = nn.Sequential(
        DepthwiseSeparableConv2d(3, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 32, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 64, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      )

      self.conv2 = nn.Sequential(
        DepthwiseSeparableConv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
      )

      self.conv3 = nn.Sequential(
        DepthwiseSeparableConv2d(64, 128, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
      )
        
    def forward(self, states):
      i1  = self.conv1(states)
      i2  = self.conv2(i1)
      i3  = self.conv3(i1)
      i23 = i2 + i3
      i4  = i23.mean([2, 3])
      return i4

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
      super(Policy_Model, self).__init__()

      self.std                  = torch.FloatTensor([1.0, 0.5, 0.5]).to(set_device(use_gpu))

      self.conv                 = CnnModel().float().to(set_device(use_gpu))
      self.memory_layer         = nn.LSTM(256, 256).float().to(set_device(use_gpu))

      self.state_extractor      = nn.Sequential( nn.Linear(2, 64), nn.ReLU() ).float().to(set_device(use_gpu))      
      self.nn_layer             = nn.Sequential( nn.Linear(320, 64), nn.ReLU() ).float().to(set_device(use_gpu))

      self.critic_layer         = nn.Sequential( nn.Linear(64, 1) ).float().to(set_device(use_gpu))
      self.actor_tanh_layer     = nn.Sequential( nn.Linear(64, 1), nn.Tanh() ).float().to(set_device(use_gpu))
      self.actor_sigmoid_layer  = nn.Sequential( nn.Linear(64, 2), nn.Sigmoid() ).float().to(set_device(use_gpu))            
        
    def forward(self, datas, detach = False):
      i   = datas[0]
      batch_size, timesteps, H, W, C  = i.shape
      
      i   = i.transpose(3, 4).transpose(2, 3).reshape(timesteps * batch_size, C, H, W)
      i   = self.conv(i)

      m         = i.reshape(timesteps, batch_size, i.shape[-1])
      m, (h, c) = self.memory_layer(m)

      s   = datas[1]
      s   = self.state_extractor(s)
      
      x   = m[-1]
      x   = torch.cat((x, s), -1)
      x   = self.nn_layer(x)

      action_tanh     = self.actor_tanh_layer(x)
      action_sigmoid  = self.actor_sigmoid_layer(x)
      action          = torch.cat((action_tanh, action_sigmoid), -1)

      if detach:
        return (action.detach(), self.std.detach()), self.critic_layer(x).detach()
      else:
        return (action, self.std), self.critic_layer(x)
      
class Value_Model(nn.Module):
    def __init__(self, state_dim, use_gpu = True):
      super(Value_Model, self).__init__()

      self.conv                 = CnnModel().float().to(set_device(use_gpu))
      self.memory_layer         = nn.LSTM(256, 256).float().to(set_device(use_gpu))

      self.state_extractor      = nn.Sequential( nn.Linear(2, 64), nn.ReLU() ).float().to(set_device(use_gpu))
      self.nn_layer             = nn.Sequential( nn.Linear(320, 64), nn.ReLU() ).float().to(set_device(use_gpu))

      self.critic_layer         = nn.Sequential( nn.Linear(64, 1) ).float().to(set_device(use_gpu))
        
    def forward(self, datas, detach = False):
      i   = datas[0]
      batch_size, timesteps, H, W, C  = datas.shape
      
      i   = datas.transpose(3, 4).transpose(2, 3).reshape(timesteps * batch_size, C, H, W)
      i   = self.conv(i)

      m         = i.reshape(timesteps, batch_size, i.shape[-1])
      m, (h, c) = self.memory_layer(m)

      s   = datas[1]
      s   = self.state_extractor(s)

      x   = m[-1]
      x   = torch.cat((x, s), -1)
      x   = self.nn_layer(x)

      if detach:
        return self.critic_layer(x).detach()
      else:
        return self.critic_layer(x)