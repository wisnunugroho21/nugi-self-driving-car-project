from model.components.SeperableConv2d import DepthwiseSeparableConv2d

import torch
import torch.nn as nn

class SpatialAtrousExtractor(nn.Module):
    def __init__(self, dim_in, dim_out, rate):
        super(SpatialAtrousExtractor, self).__init__()        

        self.spatial_atrous = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = rate, dilation = rate)
		)

    def forward(self, x):
        x = self.spatial_atrous(x)
        return x

class AtrousSpatialPyramidConv2d(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(AtrousSpatialPyramidConv2d, self).__init__()        

        self.extractor1 = SpatialAtrousExtractor(dim_in, 1, 1)
        self.extractor2 = SpatialAtrousExtractor(dim_in, 1, 4)
        self.extractor3 = SpatialAtrousExtractor(dim_in, 1, 8)

        self.out = nn.Sequential(
            DepthwiseSeparableConv2d(3, dim_out, kernel_size = 1)
        )

    def forward(self, x):
        x1 = self.extractor1(x)
        x2 = self.extractor2(x)
        x3 = self.extractor3(x) 

        xout = torch.cat((x1, x2, x3), 1)
        xout = self.out(xout)

        return xout