import numpy as np
from PIL import Image
import random
import string
import os

import numpy as np
from memory.aux_ppg.standard import auxPpgMemory

class auxPpgImageDiskMemory(auxPpgMemory):
    def __init__(self, capacity = 100000, datas = None, folder_img = '/temp/aux_ppg/'):
        self.capacity       = capacity
        self.folder_img     = folder_img

        if datas is None:
            self.states = []
        else:
            images_tens = datas
            self.states = self.__save_tensor_as_image(images_tens)

            if len(self.states) >= self.capacity:
                raise Exception('datas cannot be longer than capacity')

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        states  = self.__get_image_tens(self.states[idx])
        return np.array(states, dtype = np.float32)

    def __save_tensor_as_image(self, tensor):
        image_name  = self.folder_img + ''.join(random.choices(string.ascii_uppercase + string.digits, k = 12))
        im          = Image.fromarray(tensor)
        im.save(image_name + '.jpeg')
        
        return image_name
    
    def __get_image_tens(self, filename):
        return Image.open(filename).convert("RGB")

    def save_eps(self, state):
        if len(self) >= self.capacity:
            del self.states[0]

        self.states.append(state)

    def save_replace_all(self, states):
        self.clear_memory()
        self.save_all(states)

    def save_all(self, states):
        for state in zip(states):
            self.save_eps(state)

    def get_all_items(self):         
        return self.states

    def clear_memory(self):
        for state in self.states:
            os.remove(state)

        del self.states[:]