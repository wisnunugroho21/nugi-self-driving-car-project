import torch
from helpers.pytorch_utils import set_device

class Moco():
    def __init__(self, use_gpu):
        self.use_gpu = use_gpu

    def compute_loss(self, first_encoded, second_encoded):
        indexes     = torch.arange(first_encoded.shape[0]).long().to(set_device(self.use_gpu))   
        
        similarity  = torch.mm(first_encoded, second_encoded.t())
        return torch.nn.functional.cross_entropy(similarity, indexes)
        