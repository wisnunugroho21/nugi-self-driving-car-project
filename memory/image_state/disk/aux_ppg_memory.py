import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
import random
import string
import os

class AuxPpgMemory(Dataset):
    def __init__(self, capacity = 100000, folder_img = '/temp/'):
        self.folder_img = folder_img
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
        images  = self.trans(self.__get_image_tens(self.images[idx]))
        return np.array(self.states[idx], dtype = np.float32), images.detach().cpu().numpy()

    def __get_image_tens(self, filename):
        return Image.open(filename).convert("RGB")

    def __save_tensor_as_image(self, tensor):
        image_name  = self.folder_img + ''.join(random.choices(string.ascii_uppercase + string.digits, k = 12))
        im          = Image.fromarray(tensor)
        im.save(image_name + '.jpeg')
        
        return image_name

    def __del_image_file(self, filename):
        os.remove(filename)

    def save_eps(self, state, image, save_tensor_images = True):
        if len(self) >= self.capacity:
            del self.states[0]
            del self.images[0]

        if save_tensor_images:
            self.images.append(self.__save_tensor_as_image(image))
        else:
            self.images.append(image)

        self.states.append(state)
        self.images.append(image)

    def save_replace_all(self, states, images, save_tensor_images = True):
        self.clear_memory()
        self.save_all(states, images, save_tensor_images)

    def save_all(self, states, images, save_tensor_images = True):
        for state, image in zip(states, images):
            self.save_eps(state, image, save_tensor_images)

    def get_all_items(self, get_tensor_images = True):
        images = self.images

        if get_tensor_images:
            images = self.__get_image_tens(images)

        return self.states, self.images

    def clear_memory(self, delete_img = True):
        if delete_img:
            for state in self.states:
                self.__del_image_file(state)

        del self.states[:]
        del self.images[:]