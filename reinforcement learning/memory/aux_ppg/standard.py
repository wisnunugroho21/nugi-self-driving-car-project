import torch
from torch.utils.data import Dataset

class auxPpgMemory(Dataset):
    def __init__(self, capacity = 100000, datas = None):
        self.capacity       = capacity

        if datas is None:
            self.states         = []
        else:
            self.states = datas
            if len(self) >= self.capacity:
                raise Exception('datas cannot be longer than capacity')        

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.states[idx])

    def save_eps(self, state):
        if len(self) >= self.capacity:
            del self.states[0]

        self.states.append(state)

    def save_replace_all(self, states):
        self.clear_memory()
        self.save_all(states)

    def save_all(self, states):
        for state in states:
            self.save_eps(state)

    def get_all_items(self):         
        return self.states

    def clear_memory(self):
        del self.states[:]