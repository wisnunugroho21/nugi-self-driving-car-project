import torch

class OffPolicyLoss():
    def compute_loss(self, q_value):
        policy_loss = (q_value).mean()

        return policy_loss * -1