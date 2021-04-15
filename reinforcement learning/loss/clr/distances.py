import torch
from helpers.pytorch_utils import set_device

class DistancesClr():
    def __init__(self, use_gpu):
        self.use_gpu = use_gpu

    def compute_loss(self, first_encoded, second_encoded):
        return torch.nn.functional.pairwise_distance(second_encoded, first_encoded).mean()