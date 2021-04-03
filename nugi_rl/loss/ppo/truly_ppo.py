import torch

class TrulyPPO():
    def __init__(self, distribution, advantage_function, policy_kl_range = 0.0008, policy_params = 20, value_clip = 1.0, vf_loss_coef = 1.0, entropy_coef = 0.01, gamma = 0.95):
        self.policy_kl_range    = policy_kl_range
        self.policy_params      = policy_params
        self.value_clip         = value_clip
        self.vf_loss_coef       = vf_loss_coef
        self.entropy_coef       = entropy_coef

        self.advantage_function = advantage_function
        self.distribution       = distribution

    def compute_loss(self, action_datas, old_action_datas, values, old_values, next_values, actions, rewards, dones):
        advantages      = self.advantage_function.compute_advantages(rewards, values, next_values, dones)
        returns         = (advantages + values).detach()
        advantages      = ((advantages - advantages.mean()) / (advantages.std() + 1e-6)).detach()       

        logprobs        = self.distribution.logprob(action_datas, actions) + 1e-5
        old_logprobs    = (self.distribution.logprob(old_action_datas, actions) + 1e-5).detach()

        ratios          = (logprobs - old_logprobs).exp()       
        Kl              = self.distribution.kldivergence(old_action_datas, action_datas) + 1e-5

        pg_targets  = torch.where(
            (Kl >= self.policy_kl_range) & (ratios > 1),
            ratios * advantages - self.policy_params * Kl,
            ratios * advantages
        )
        pg_loss         = pg_targets.mean()
        dist_entropy    = self.distribution.entropy(action_datas).mean()

        if self.value_clip is None:
            critic_loss     = ((returns - values).pow(2) * 0.5).mean()
        else:
            vpredclipped    = old_values + torch.clamp(values - old_values, -self.value_clip, self.value_clip)
            critic_loss     = ((returns - vpredclipped).pow(2) * 0.5).mean()

        loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss