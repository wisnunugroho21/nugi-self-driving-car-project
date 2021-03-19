class ValueFunction():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma

    def compute_value(self, reward, next_value, done):
        q_values = reward + (1.0 - done) * self.gamma * next_value           
        return q_values