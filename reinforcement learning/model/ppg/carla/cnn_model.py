import torch.nn as nn

from model.components.ASPP import AtrousSpatialPyramidConv2d
from model.components.SeperableConv2d import DepthwiseSeparableConv2d
from model.components.Downsampler import Downsampler

class CnnModel(nn.Module):
    def __init__(self):
      super(CnnModel, self).__init__() 

      self.conv = nn.Sequential(
        AtrousSpatialPyramidConv2d(3, 8),
        nn.ReLU(),
        DepthwiseSeparableConv2d(8, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        Downsampler(16, 16),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(32, 32, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        Downsampler(32, 32),
        nn.ReLU(),
        DepthwiseSeparableConv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        Downsampler(64, 64),
        nn.ReLU(),
        DepthwiseSeparableConv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 128, kernel_size = 5, stride = 1, padding = 0),
        nn.ReLU(),
      )
        
    def forward(self, image, detach = False):
      out = self.conv(image)
      out = out.mean([-1, -2])

      if detach:
        return out.detach()
      else:
        return out