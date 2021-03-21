import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

class PolicyMemory(Dataset):
    def __init__(self, capacity = 100000):
        self.capacity       = capacity

        self.states         = []
        self.images         = []
        self.actions        = []
        self.rewards        = []
        self.dones          = []
        self.next_states    = []
        self.next_images    = []

        self.trans  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        images      = self.trans(self.images[idx])
        next_images = self.trans(self.next_images[idx])

        return np.array(self.states[idx], dtype = np.float32), images.detach().cpu().numpy(), np.array(self.actions[idx], dtype = np.float32), \
            np.array([self.rewards[idx]], dtype = np.float32), np.array([self.dones[idx]], dtype = np.float32), \
            np.array(self.next_states[idx], dtype = np.float32), next_images.detach().cpu().numpy()       

    def save_eps(self, state, image, action, reward, done, next_state, next_image):
        if len(self) >= self.capacity:
            del self.states[0]
            del self.images[0]
            del self.actions[0]
            del self.rewards[0]
            del self.dones[0]
            del self.next_states[0]
            del self.next_images[0]

        self.states.append(state)
        self.images.append(image)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.next_images.append(next_image)

    def save_replace_all(self, states, images, actions, rewards, dones, next_states, next_images):
        self.clear_memory()
        self.save_all(states, images, actions, rewards, dones, next_states, next_images)

    def save_all(self, states, images, actions, rewards, dones, next_states, next_images):
        for state, image, action, reward, done, next_state, next_image in zip(states, images, actions, rewards, dones, next_states, next_images):
            self.save_eps(state, image, action, reward, done, next_state, next_image)

    def get_all_items(self):         
        return self.states, self.images, self.actions, self.rewards, self.dones, self.next_states, self.next_images

    def clear_memory(self):
        del self.states[:]
        del self.images[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]
        del self.next_images[:]