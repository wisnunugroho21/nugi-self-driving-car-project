import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
import random
import string
import os

class AuxClrMemory(Dataset):
    def __init__(self, capacity = 10000, first_trans = None, second_trans = None, folder_img = '/temp/'):
        self.folder_img     = folder_img        
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
        first_inputs    = self.first_trans(self.__get_image_tens(self.images[idx]))
        second_inputs   = self.second_trans(self.__get_image_tens(self.images[idx]))

        return (first_inputs.detach().cpu().numpy(), second_inputs.detach().cpu().numpy())

    def __get_image_tens(self, filename):
        return Image.open(filename).convert("RGB")

    def __save_tensor_as_image(self, tensor):
        image_name  = self.folder_img + ''.join(random.choices(string.ascii_uppercase + string.digits, k = 12))
        im          = Image.fromarray(tensor)
        im.save(image_name + '.jpeg')
        
        return image_name

    def __del_image_file(self, filename):
        os.remove(filename)

    def save_eps(self, image, save_tensor_images = True):
        if len(self) >= self.capacity:
            del self.images[0]

        if save_tensor_images:
            self.images.append(self.__save_tensor_as_image(image))
        else:
            self.images.append(image)

        self.images.append(image)

    def save_replace_all(self, images, save_tensor_images = True):
        self.clear_memory()
        self.save_all(images, save_tensor_images)

    def save_all(self, images, save_tensor_images = True):
        for image in images:
            self.save_eps(image, save_tensor_images)

    def get_all_items(self, get_tensor_images = True):  
        images = self.images

        if get_tensor_images:
            images = self.__get_image_tens(images)

        return self.images

    def clear_memory(self, delete_img = True):
        if delete_img:
            for state in self.states:
                self.__del_image_file(state)
                
        del self.images[:]

    def save_redis(self):
        self.redis.append('images', self.images)

    def load_redis(self):
        self.images = self.redis.lrange('images', 0, -1)

    def delete_redis(self):
        self.redis.delete('images')