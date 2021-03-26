import torch.nn as nn
import torch

from model.component.atrous_spatial_pyramid_conv2d import AtrousSpatialPyramidConv2d
from model.component.depthwise_separable_conv2d import DepthwiseSeparableConv2d

class CnnModel(nn.Module):
    def __init__(self):
      super(CnnModel, self).__init__()   

      self.conv = nn.Sequential(
        AtrousSpatialPyramidConv2d(21, 7, 21),
        nn.ReLU(),
        DepthwiseSeparableConv2d(21, 8, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(8, 16, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 32, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
        DepthwiseSeparableConv2d(32, 64, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
        DepthwiseSeparableConv2d(64, 128, kernel_size = 5, stride = 1, padding = 0),
        nn.ReLU(),
      )
        
    def forward(self, image, detach = False):
      image = torch.nn.functional.one_hot(image, num_classes = 21).transpose(2, 3).transpose(1, 2).float()
      
      out = self.conv(image)
      out = out.mean([-1, -2])

      if detach:
        return out.detach()
      else:
        return out