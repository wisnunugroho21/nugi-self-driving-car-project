class Cql():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma

    def compute_loss(self, naive_predicted_q_value, predicted_q_value, reward, done, next_value):
        cql_regularizer = naive_predicted_q_value - predicted_q_value

        target_q_value  = (reward + (1 - done) * self.gamma * next_value).detach()
        td_error        = ((target_q_value - predicted_q_value).pow(2) * 0.5).mean()

        return td_error + cql_regularizer