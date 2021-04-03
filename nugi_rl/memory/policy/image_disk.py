import numpy as np
from PIL import Image
import random
import string
import os

import numpy as np
from memory.policy.standard import PolicyMemory

class ImagePolicyMemory(PolicyMemory):
    def __init__(self, capacity = 100000, datas = None, folder_img = '/temp/'):
        self.capacity       = capacity
        self.position       = 0
        self.folder_img     = folder_img

        if datas is None:
            self.states         = []
            self.actions        = []
            self.rewards        = []
            self.dones          = []
            self.next_states    = []
        else:
            images_tens, self.actions, self.rewards, self.dones, next_images_tens = datas

            self.states         = []
            self.next_states    = []

            for image_tens, next_image_tens in zip(images_tens, next_images_tens):
                self.states.append(self.__save_tensor_as_image(image_tens))
                self.next_states.append(self.__save_tensor_as_image(next_image_tens))

            if len(self.dones) >= self.capacity:
                raise Exception('datas cannot be longer than capacity')

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        states      = self.__get_image_tens(self.states[idx])
        next_states = self.__get_image_tens(self.next_states[idx])

        return np.array(states, dtype = np.float32), np.array(self.actions[idx], dtype = np.float32), \
            np.array([self.rewards[idx]], dtype = np.float32), np.array([self.dones[idx]], dtype = np.float32), \
            np.array(next_states, dtype = np.float32)

    def __save_tensor_as_image(self, tensor):
        image_name  = self.folder_img + ''.join(random.choices(string.ascii_uppercase + string.digits, k = 12))
        im          = Image.fromarray(tensor)
        im.save(image_name + '.jpeg')
        
        return image_name
    
    def __get_image_tens(self, filename):
        return Image.open(filename).convert("RGB")

    def save_eps(self, state, action, reward, done, next_state, save_tensor_images = True):
        if len(self) >= self.capacity:
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.dones[0]
            del self.next_states[0]

        if save_tensor_images:
            self.states.append(self.__save_tensor_as_image(state))
            self.next_states.append(self.__save_tensor_as_image(next_state))
        else:
            self.states.append(state)
            self.next_states.append(next_state)
        
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)        

    def save_replace_all(self, states, actions, rewards, dones, next_states, save_img = True, delete_img = True):
        self.clear_memory(delete_img)
        self.save_all(states, actions, rewards, dones, next_states, save_img)

    def save_all(self, states, actions, rewards, dones, next_states, save_img = True):
        for state, action, reward, done, next_state in zip(states, actions, rewards, dones, next_states):
            self.save_eps(state, action, reward, done, next_state, save_img)

    def get_all_items(self, get_tensor_images = True):
        states = self.states
        next_states = self.next_states

        if get_tensor_images:
            states = self.__get_image_tens(self.states)
            next_states = self.__get_image_tens(self.next_states)

        return states, self.actions, self.rewards, self.dones, next_states 

    def clear_memory(self, delete_img = True):
        if delete_img:
            for state in self.states:
                os.remove(state)

            for next_state in self.next_states:
                os.remove(next_state)

        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]

    def clear_idx(self, idx, delete_img = True):
        if delete_img:
            os.remove(self.states[idx])
            os.remove(self.next_states[idx])

        del self.states[idx]
        del self.actions[idx]
        del self.rewards[idx]
        del self.dones[idx]
        del self.next_states[idx]