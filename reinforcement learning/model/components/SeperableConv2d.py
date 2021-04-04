import torch
import torch.nn as nn
from helpers.pytorch_utils import set_device

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, stride = 1, padding = 0, dilation = 1, bias = True, depth_multiplier = 1):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        self.nn_layer = nn.Sequential(
            nn.Conv2d(nin, nin * depth_multiplier, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias, groups = nin),
            nn.Conv2d(nin * depth_multiplier, nout, kernel_size = 1, bias = bias)
        )

    def forward(self, x):
        return self.nn_layer(x)

class SeparableConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, stride = 1, padding = 0, dilation = 1, depth_multiplier = 1):
        super(SeparableConv2d, self).__init__()
        
        self.nn_layer = nn.Sequential(
            nn.Conv2d(nin, nout * depth_multiplier, kernel_size = 1),
            nn.Conv2d(nout * depth_multiplier, nout, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = nout)            
        )

    def forward(self, x):
        return self.nn_layer(x)