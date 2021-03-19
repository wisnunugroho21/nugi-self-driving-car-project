import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

class AuxClrMemory(Dataset):
    def __init__(self, capacity = 10000, first_trans = None, second_trans = None):        
        self.images         = []
        self.capacity       = capacity
        self.first_trans    = first_trans
        self.second_trans   = second_trans

        if self.first_trans is None:
            self.first_trans = transforms.Compose([
                transforms.RandomResizedCrop(320),                           
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p = 0.8),
                transforms.RandomGrayscale(p = 0.2),
                transforms.GaussianBlur(33),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if self.second_trans is None:
            self.second_trans = transforms.Compose([     
                transforms.RandomResizedCrop(320),                           
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p = 0.8),
                transforms.RandomGrayscale(p = 0.2),
                transforms.GaussianBlur(33),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images          = self.images[idx] # torch.FloatTensor(self.images[idx])

        first_inputs    = self.first_trans(images)
        second_inputs   = self.second_trans(images)

        return (first_inputs.detach().cpu().numpy(), second_inputs.detach().cpu().numpy())

    def save_eps(self, image):
        if len(self) >= self.capacity:
            del self.images[0]

        self.images.append(image)

    def save_replace_all(self, images):
        self.clear_memory()

        for image in images:
            self.save_eps(image)

    def save_all(self, images):
        for image in images:
            self.save_eps(image)

    def get_all_items(self):         
        return self.images

    def clear_memory(self):
        del self.images[:]