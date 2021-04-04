import numpy as np
from torch.utils.data import Dataset

class PolicyMemory(Dataset):
    def __init__(self, capacity = 100000, datas = None):
        self.capacity       = capacity
        self.position       = 0

        if datas is None:
            self.states         = []
            self.actions        = []
            self.rewards        = []
            self.dones          = []
            self.next_states    = []
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states = datas
            if len(self.dones) >= self.capacity:
                raise Exception('datas cannot be longer than capacity')        

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32), np.array(self.actions[idx], dtype = np.float32), \
            np.array([self.rewards[idx]], dtype = np.float32), np.array([self.dones[idx]], dtype = np.float32), \
            np.array(self.next_states[idx], dtype = np.float32)      

    def save_eps(self, state, action, reward, done, next_state):
        if len(self) >= self.capacity:
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.dones[0]
            del self.next_states[0]

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def save_replace_all(self, states, actions, rewards, dones, next_states):
        self.clear_memory()
        self.save_all(states, actions, rewards, dones, next_states)

    def save_all(self, states, actions, rewards, dones, next_states):
        for state, action, reward, done, next_state in zip(states, actions, rewards, dones, next_states):
            self.save_eps(state, action, reward, done, next_state)

    def get_all_items(self):         
        return self.states, self.actions, self.rewards, self.dones, self.next_states 

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]