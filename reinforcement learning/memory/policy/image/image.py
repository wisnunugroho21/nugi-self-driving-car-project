import torchvision.transforms as transforms
import torch
from memory.policy.standard import PolicyMemory

class ImagePolicyMemory(PolicyMemory):
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

        self.trans  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        states      = self.trans(self.states[idx])
        next_states = self.trans(self.next_states[idx])

        return states, torch.FloatTensor(self.actions[idx]), torch.FloatTensor([self.rewards[idx]]), \
            torch.FloatTensor([self.dones[idx]]), next_states