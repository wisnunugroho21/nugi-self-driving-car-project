import torch

class OffPolicyLoss():
    def compute_loss(self, q_value1, q_value2):
        new_q_value = torch.min(q_value1, q_value2)
        policy_loss = (new_q_value).mean()

        return policy_loss * -1