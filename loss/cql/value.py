import torch

class OffVLoss():
    def compute_loss(self, predicted_value, q_value):
        target_value    = q_value.detach()
        value_loss      = ((target_value - predicted_value).pow(2) * 0.5).mean()

        return value_loss