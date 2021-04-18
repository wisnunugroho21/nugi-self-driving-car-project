import torch.nn as nn

from model.components.ASPP import AtrousSpatialPyramidConv2d
from model.components.SeperableConv2d import DepthwiseSeparableConv2d

class CnnModel(nn.Module):
    def __init__(self):
      super(CnnModel, self).__init__()   

      self.bn1 = nn.BatchNorm2d(16)
      self.bn2 = nn.BatchNorm2d(32)
      self.bn3 = nn.BatchNorm2d(64)

      self.conv1 = nn.Sequential(
        AtrousSpatialPyramidConv2d(3, 8),
        nn.ReLU(),
        DepthwiseSeparableConv2d(8, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
      )

      self.conv2 = nn.Sequential(
        DepthwiseSeparableConv2d(16, 16, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
        DepthwiseSeparableConv2d(16, 16, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
      )

      self.conv3 = nn.Sequential(
        DepthwiseSeparableConv2d(16, 16, kernel_size = 8, stride = 4, padding = 2, bias = False),
        nn.ReLU(),
      )

      self.conv4 = nn.Sequential(
        DepthwiseSeparableConv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(32, 32, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
      )

      self.conv5 = nn.Sequential(
        DepthwiseSeparableConv2d(32, 32, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
        DepthwiseSeparableConv2d(32, 32, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
      )

      self.conv6 = nn.Sequential(
        DepthwiseSeparableConv2d(32, 32, kernel_size = 8, stride = 4, padding = 2, bias = False),
        nn.ReLU(),
      )

      self.conv7 = nn.Sequential(
        DepthwiseSeparableConv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
      )

      self.conv8 = nn.Sequential(
        DepthwiseSeparableConv2d(64, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
        DepthwiseSeparableConv2d(64, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
      )

      self.conv9 = nn.Sequential(
        DepthwiseSeparableConv2d(64, 64, kernel_size = 8, stride = 4, padding = 2, bias = False),
        nn.ReLU(),
      )

      self.conv_out = nn.Sequential(
        DepthwiseSeparableConv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        DepthwiseSeparableConv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
      )
        
    def forward(self, image, detach = False):
      i1  = self.conv1(image)
      i2  = self.conv2(i1)
      i3  = self.conv3(i1)
      i4  = self.conv4(self.bn1(i2 + i3))
      i5  = self.conv5(i4)
      i6  = self.conv6(i4)
      i7  = self.conv7(self.bn2(i5 + i6))
      i8  = self.conv8(i7)
      i9  = self.conv9(i7)
      out = self.conv_out(self.bn3(i8 + i9))
      out = out.mean([-1, -2])

      if detach:
        return out.detach()
      else:
        return out