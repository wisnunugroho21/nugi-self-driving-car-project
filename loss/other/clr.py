import torch
from helper.pytorch import set_device

class CLR():
    def __init__(self, use_gpu):
        self.use_gpu = use_gpu

    def compute_loss(self, first_encoded, second_encoded):
        indexes     = torch.arange(first_encoded.shape[0]).long().to(set_device(self.use_gpu))   
        
        similarity  = torch.nn.functional.cosine_similarity(first_encoded.unsqueeze(1), second_encoded.unsqueeze(0), dim = 2)
        return torch.nn.functional.cross_entropy(similarity, indexes)