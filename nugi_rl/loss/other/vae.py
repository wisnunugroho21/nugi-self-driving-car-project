class VAE():
    def compute_loss(self, states, reconstruc_states):
        dif_states  = ((states - reconstruc_states).pow(2) * 0.5).mean()
        return dif_states