import torch
import torch.nn as nn

from model.component.spatial_atrous_extractor import SpatialAtrousExtractor

class AtrousSpatialPyramidConv2d(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(AtrousSpatialPyramidConv2d, self).__init__()        

        self.extractor1 = SpatialAtrousExtractor(dim_in, 1)
        self.extractor2 = SpatialAtrousExtractor(dim_in, 3)
        self.extractor3 = SpatialAtrousExtractor(dim_in, 6)

        self.out = nn.Sequential(
            nn.Conv2d(3 * dim_in, dim_out, kernel_size = 1)
        )

    def forward(self, x):
        x1 = self.extractor1(x)
        x2 = self.extractor2(x)
        x3 = self.extractor3(x) 

        xout = torch.cat((x1, x2, x3), 1)
        xout = self.out(xout)

        return xout