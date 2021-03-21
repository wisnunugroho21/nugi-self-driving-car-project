import torch.nn as nn

from model.component.atrous_spatial_pyramid_conv2d import AtrousSpatialPyramidConv2d
from model.component.depthwise_separable_conv2d import DepthwiseSeparableConv2d

class CnnModel(nn.Module):
    def __init__(self):
      super(CnnModel, self).__init__()   

      self.conv = nn.Sequential(
        AtrousSpatialPyramidConv2d(21, 21),
        nn.ReLU(),        
        DepthwiseSeparableConv2d(21, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 32, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
        DepthwiseSeparableConv2d(32, 64, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
        DepthwiseSeparableConv2d(64, 128, kernel_size = 8, stride = 4, padding = 2),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
      )
        
    def forward(self, image, detach = False):
      out = self.conv(image)

      if detach:
        return out.detach()
      else:
        return out