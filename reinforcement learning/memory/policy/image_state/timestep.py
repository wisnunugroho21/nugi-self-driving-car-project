import torch
import torchvision.transforms as transforms
from memory.policy.standard import PolicyMemory

class TimestepISPMemory(PolicyMemory):
    def __init__(self, datas = None):
        if datas is None :
            self.images         = []
            self.next_images    = []
            super().__init__()

        else:
            data_states, actions, rewards, dones, next_data_states = datas
            images, states              = data_states
            next_images, next_states    = next_data_states

            self.images                 = images
            self.next_images            = next_images
            
            super().__init__((states, actions, rewards, dones, next_states))

        self.trans  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        images      = torch.stack([self.trans(image) for image in self.images[idx]])
        next_images = torch.stack([self.trans(next_image) for next_image in self.next_images[idx]])

        states, actions, rewards, dones, next_states = super().__getitem__(idx)
        return images, states, actions, rewards, dones, next_images, next_states

    def save_eps(self, image, state, action, reward, done, next_image, next_state):
        if len(self) >= self.capacity:
            del self.images[0]
            del self.next_images[0]

        super().save_eps(state, action, reward, done, next_state)
        self.images.append(image)
        self.next_images.append(next_image)

    def save_replace_all(self, images, states, actions, rewards, dones, next_images, next_states):
        self.clear_memory()
        self.save_all(images, states, actions, rewards, dones, next_images, next_states)

    def save_all(self, images, states, actions, rewards, dones, next_images, next_states):
        for image, state, action, reward, done, next_image, next_state in zip(images, states, actions, rewards, dones, next_images, next_states):            
            self.save_eps(image, state, action, reward, done, next_image, next_state)

    def get_all_items(self):
        states, actions, rewards, dones, next_states = super().get_all_items()
        return self.images, states, actions, rewards, dones, self.next_images, next_states

    def clear_memory(self):
        super().clear_memory()
        del self.images[:]
        del self.next_images[:]

    def transform(self, images):
        return torch.stack([self.trans(image) for image in images])
