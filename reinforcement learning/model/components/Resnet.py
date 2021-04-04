import torch
import torch.nn as nn
from helpers.pytorch_utils import set_device

class Resnet(nn.Module):
  def __init__(self, use_gpu = True):
    super(Resnet, self).__init__() 

    self.input_layer_1 = nn.Sequential(
      nn.Conv2d(1, 8, kernel_size = 4, stride = 2, padding = 1))

    self.input_layer_1_1 = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(8, 8, kernel_size = 3, stride = 1, padding = 1))

    self.input_layer_2 = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(8, 8, kernel_size = 4, stride = 2, padding = 1))

    self.input_layer_2_1 = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(8, 8, kernel_size = 3, stride = 1, padding = 1))

    self.input_layer_3 = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(8, 8, kernel_size = 4, stride = 2, padding = 1))

    self.input_layer_3_1 = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(8, 8, kernel_size = 3, stride = 1, padding = 1))

    self.input_layer_4 = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(8, 8, kernel_size = 4, stride = 2, padding = 1))

    self.input_layer_4_1 = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(8, 8, kernel_size = 3, stride = 1, padding = 1))

    self.input_post_layer = nn.Sequential(
      nn.ReLU(),
      nn.Flatten(),        
      nn.Linear(800, 200),
      nn.ReLU())

  def forward(self, states):  
      x         = states

      x         = self.input_layer_1(x)
      x1        = self.input_layer_1_1(x)
      x         = torch.add(x, x1)

      x         = self.input_layer_2(x)
      x1        = self.input_layer_2_1(x)
      x         = torch.add(x, x1)

      x         = self.input_layer_3(x)
      x1        = self.input_layer_3_1(x)
      x         = torch.add(x, x1)

      x         = self.input_layer_4(x)
      x1        = self.input_layer_4_1(x)
      x         = torch.add(x, x1)

      x         = self.input_post_layer(x)
      return x