import numpy as np

from memory.policy.standard import PolicyMemory

class EmbeddingPolicyMemory(PolicyMemory):
    def __init__(self, datas):
        if datas is None :
            self.available_actions = []
            super().__init__()

        else:
            states, actions, rewards, dones, next_states, available_actions = datas
            self.available_actions = available_actions
            
            super().__init__((states, actions, rewards, dones, next_states))

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        states, actions, rewards, dones, next_states = super().__getitem__(idx)
        return states, actions, rewards, dones, next_states, np.array(self.available_actions[idx], dtype = np.float32)      

    def save_eps(self, state, action, reward, done, next_state, available_action):
        super().save_eps(state, reward, action, done, next_state)
        self.available_actions.append(available_action)

    def save_replace_all(self, states, actions, rewards, dones, next_states, available_actions):
        super().save_all(states, rewards, actions, dones, next_states)
        self.available_actions = available_actions

    def save_all(self, states, actions, rewards, dones, next_states, available_actions):
        super().save_all(states, actions, rewards, dones, next_states)
        self.available_actions    += available_actions

    def get_all_items(self):
        states, actions, rewards, dones, next_states = super().get_all_items()
        return states, actions, rewards, dones, next_states, self.available_actions

    def clear_memory(self):
        super().clear_memory()
        del self.available_actions[:]
