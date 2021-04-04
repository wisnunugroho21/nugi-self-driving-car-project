import torch
from helpers.pytorch_utils import set_device

class ValueFunction():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma

    def compute_value(self, reward, done):
        returns = []        
        running_add = 0
        
        for i in reversed(range(len(reward))):
            running_add = reward[i] + (1.0 - done) * self.gamma * running_add  
            returns.insert(0, running_add)
            
        return torch.stack(returns)
        
    def temporal_difference(self, reward, next_value, done):
        q_values = reward + (1.0 - done) * self.gamma * next_value           
        return q_values