import torch.nn as nn

class SpatialAtrousExtractor(nn.Module):
    def __init__(self, dim, rate):
        super(SpatialAtrousExtractor, self).__init__()        

        self.spatial_atrous = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = 3, stride = 1, padding = rate, dilation = rate, bias = False, groups = dim)
		)

    def forward(self, x):
        x = self.spatial_atrous(x)
        return x