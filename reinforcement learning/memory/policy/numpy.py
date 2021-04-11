import numpy as np
import torch

from memory.policy.standard import PolicyMemory

class NumpyPolicyMemory(PolicyMemory):
    def __init__(self, datas = None):
        if datas is None :
            self.states         = np.array([], dtype = np.float32)
            self.actions        = np.array([], dtype = np.float32)
            self.rewards        = np.array([], dtype = np.float32)
            self.dones          = np.array([], dtype = np.float32)
            self.next_states    = np.array([], dtype = np.float32)
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states = datas

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return torch.from_numpy(self.states[idx]), torch.from_numpy(self.actions[idx]), torch.from_numpy(self.rewards[idx]).unsqueeze(1), \
            torch.from_numpy(self.dones[idx]).unsqueeze(1), torch.from_numpy(self.next_states[idx])     

    def save_eps(self, state, action, reward, done, next_state):
        if len(self) == 0:
            self.states         = np.array([state], dtype = np.float32)
            self.actions        = np.array([action], dtype = np.float32)
            self.rewards        = np.array([[reward]], dtype = np.float32)
            self.dones          = np.array([[done]], dtype = np.float32)
            self.next_states    = np.array([next_state], dtype = np.float32)

        else:
            self.states         = np.append(self.states, [state], axis = 0)
            self.actions        = np.append(self.actions, [action], axis = 0)
            self.rewards        = np.append(self.rewards, [[reward]], axis = 0)
            self.dones          = np.append(self.dones, [[done]], axis = 0)
            self.next_states    = np.append(self.next_states, [next_state], axis = 0)

    def save_replace_all(self, states, actions, rewards, dones, next_states):
        self.states         = np.array(states)
        self.actions        = np.array(actions)
        self.rewards        = np.array(rewards)
        self.dones          = np.array(dones)
        self.next_states    = np.array(next_states)

    def save_all(self, states, actions, rewards, dones, next_states):
        self.states         = np.concatenate(np.array(states))
        self.actions        = np.concatenate(np.array(actions))
        self.rewards        = np.concatenate(np.array(rewards))
        self.dones          = np.concatenate(np.array(dones))
        self.next_states    = np.concatenate(np.array(next_states))

    def get_all_items(self):         
        return self.states, self.actions, self.rewards, self.dones, self.next_states

    def clear_memory(self):
        self.states         = np.delete(self.states, np.s_[:])
        self.actions        = np.delete(self.actions, np.s_[:])
        self.rewards        = np.delete(self.rewards, np.s_[:])
        self.dones          = np.delete(self.dones, np.s_[:])
        self.next_states    = np.delete(self.next_states, np.s_[:])
