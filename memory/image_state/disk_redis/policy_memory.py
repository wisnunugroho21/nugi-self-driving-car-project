import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
import random
import string
import os

class PolicyMemory(Dataset):
    def __init__(self, redis, capacity = 100000, folder_img = '/temp/'):
        self.redis          = redis
        self.folder_img     = folder_img
        self.capacity       = capacity
        self.position       = 0

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
        images      = self.trans(self.__get_image_tens(self.images[idx]))
        next_images = self.trans(self.__get_image_tens(self.next_images[idx]))

        return np.array(self.states[idx], dtype = np.float32), images.detach().cpu().numpy(), np.array(self.actions[idx], dtype = np.float32), \
            np.array([self.rewards[idx]], dtype = np.float32), np.array([self.dones[idx]], dtype = np.float32), \
            np.array(self.next_states[idx], dtype = np.float32), next_images.detach().cpu().numpy()

    def __get_image_tens(self, filename):
        return Image.open(filename).convert('RGB')

    def __save_tensor_as_image(self, tensor):
        image_name  = self.folder_img + ''.join(random.choices(string.ascii_uppercase + string.digits, k = 12))
        im          = Image.fromarray(tensor)
        im.save(image_name + '.jpeg')
        
        return image_name

    def __del_image_file(self, filename):
        os.remove(filename)

    def save_eps(self, state, image, action, reward, done, next_state, next_image, save_tensor_images = True):
        if len(self) >= self.capacity:
            del self.states[0]
            del self.images[0]
            del self.actions[0]
            del self.rewards[0]
            del self.dones[0]
            del self.next_states[0]
            del self.next_images[0]

        if save_tensor_images:
            self.images.append(self.__save_tensor_as_image(image))
            self.next_images.append(self.__save_tensor_as_image(next_image))
        else:
            self.images.append(image)
            self.next_images.append(next_image)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def save_replace_all(self, states, images, actions, rewards, dones, next_states, next_images, save_tensor_images = True):
        self.clear_memory()
        self.save_all(states, images, actions, rewards, dones, next_states, next_images, save_tensor_images)

    def save_all(self, states, images, actions, rewards, dones, next_states, next_images, save_tensor_images = True):
        for state, image, action, reward, done, next_state, next_image in zip(states, images, actions, rewards, dones, next_states, next_images):
            self.save_eps(state, image, action, reward, done, next_state, next_image, save_tensor_images)

    def get_all_items(self, get_tensor_images = True): 
        images = self.images
        next_images = self.next_images

        if get_tensor_images:
            images = self.__get_image_tens(images)
            next_images = self.__get_image_tens(next_images)

        return self.states, images, self.actions, self.rewards, self.dones, self.next_states, next_images

    def clear_memory(self, delete_img = True):
        if delete_img:
            for state in self.states:
                self.__del_image_file(state)

            for next_state in self.next_states:
                self.__del_image_file(next_state)

        del self.states[:]
        del self.images[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]
        del self.next_images[:]

    def save_redis(self):
        self.redis.append('states', self.states)
        self.redis.append('images', self.images)
        self.redis.append('actions', self.actions)
        self.redis.append('rewards', self.rewards)
        self.redis.append('dones', self.dones)
        self.redis.append('next_states', self.next_states)
        self.redis.append('next_images', self.next_images)

    def load_redis(self):
        self.states         = self.redis.lrange('states', 0, -1)
        self.images         = self.redis.lrange('images', 0, -1)
        self.actions        = self.redis.lrange('actions', 0, -1)
        self.rewards        = self.redis.lrange('rewards', 0, -1)
        self.dones          = self.redis.lrange('dones', 0, -1)
        self.next_states    = self.redis.lrange('next_states', 0, -1)
        self.next_images    = self.redis.lrange('next_images', 0, -1)

    def delete_redis(self):
        self.redis.delete('states')
        self.redis.delete('images')
        self.redis.delete('actions')
        self.redis.delete('rewards')
        self.redis.delete('dones')
        self.redis.delete('next_states')
        self.redis.delete('next_images')