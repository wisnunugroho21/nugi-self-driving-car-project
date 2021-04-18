import torch
from helpers.pytorch_utils import set_device

class CrossEntropyIndexes():
    def __init__(self, use_gpu):
        self.use_gpu = use_gpu

    def compute_loss(self, logits):
        indexes = torch.arange(logits.shape[0]).long().to(set_device(self.use_gpu))
        return torch.nn.functional.cross_entropy(logits, indexes)
        