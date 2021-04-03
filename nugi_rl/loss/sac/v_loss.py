import torch

class VLoss():
    def __init__(self, distribution):
        self.distribution       = distribution

    def compute_loss(self, predicted_value, action_datas, actions, q_value1, q_value2):
        log_prob                = self.distribution.logprob(action_datas, actions)
        q_value                 = torch.min(q_value1, q_value2)
        target_value            = (q_value - log_prob).detach()

        value_loss              = ((target_value - predicted_value).pow(2) * 0.5).mean()
        return value_loss