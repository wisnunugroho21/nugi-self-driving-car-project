import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

class AuxPpgMemory(Dataset):
    def __init__(self, capacity = 100000):
        self.capacity   = capacity
        self.images     = []
        self.states     = []

        self.trans  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])    

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        images  = self.trans(self.images[idx])
        return np.array(self.states[idx], dtype = np.float32), images.detach().cpu().numpy()

    def save_eps(self, state, image):
        if len(self) >= self.capacity:
            del self.states[0]
            del self.images[0]

        self.states.append(state)
        self.images.append(image)

    def save_replace_all(self, states, images):
        self.clear_memory()
        self.save_all(states, images)

    def save_all(self, states, images):
        for state, image in zip(states, images):
            self.save_eps(state, image)

    def get_all_items(self):         
        return self.states, self.images

    def clear_memory(self):
        del self.states[:]
        del self.images[:]