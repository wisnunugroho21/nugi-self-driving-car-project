from torch.distributions import Categorical, Normal
from torch.distributions.kl import kl_divergence
import torch

from helpers.pytorch_utils import set_device, to_numpy


class BasicContinous():
    def __init__(self, use_gpu):
        self.use_gpu = use_gpu

    # def sample(self, datas):
    #     mean, std = datas

    #     distribution    = Normal(0, 1)
    #     rand            = distribution.sample().float().to(set_device(self.use_gpu))
    #     return (mean + std * rand).squeeze(0)

    def sample(self, datas):
        mean, std = datas

        distribution = Normal(mean, std.squeeze())
        return distribution.sample().float().to(set_device(self.use_gpu)).squeeze(0)
        
    def entropy(self, datas):
        mean, std = datas
        
        distribution = Normal(mean, std)
        return distribution.entropy().float().to(set_device(self.use_gpu))
        
    def logprob(self, datas, value_data):
        mean, std = datas

        distribution = Normal(mean, std)
        return distribution.log_prob(value_data).float().to(set_device(self.use_gpu))

    def kldivergence(self, datas1, datas2):
        mean1, std1 = datas1
        mean2, std2 = datas2

        distribution1 = Normal(mean1, std1)
        distribution2 = Normal(mean2, std2)
        return kl_divergence(distribution1, distribution2).float().to(set_device(self.use_gpu))

    def deterministic(self, datas):
        mean, _ = datas
        return mean.squeeze(0)