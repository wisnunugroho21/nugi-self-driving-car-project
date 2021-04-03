class JointAux():
    def __init__(self, distribution):
        self.distribution       = distribution

    def compute_loss(self, action_datas, old_action_datas, values, returns):
        Kl                  = self.distribution.kldivergence(old_action_datas, action_datas).mean()
        aux_loss            = ((returns - values).pow(2) * 0.5).mean()

        return aux_loss + Kl