import torch

class PolicyLoss():
    def __init__(self, distribution):
        self.distribution       = distribution

    def compute_loss(self, action_datas, actions, predicted_q_value1, predicted_q_value2):
        log_prob                = self.distribution.logprob(action_datas, actions)
        predicted_new_q_value   = torch.min(predicted_q_value1, predicted_q_value2)

        policy_loss             = (predicted_new_q_value - log_prob).mean()
        return policy_loss * -1