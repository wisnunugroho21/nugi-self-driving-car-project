import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
import numpy as np
from PIL import Image

class CarlaDataset(Dataset):
    def __init__(self, root = '', input_trans = None, target_trans = None ):
        self.root           = root
        self.input_trans    = input_trans
        self.target_trans   = target_trans
        self.imgs           = list(sorted(os.listdir(os.path.join(self.root, ''))))

        if self.input_trans is None:
            self.input_trans = transforms.Compose([
                # transforms.RandomResizedCrop(320),                           
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p = 0.8),
                transforms.RandomGrayscale(p = 0.2),
                transforms.GaussianBlur(33),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if self.target_trans is None:
            self.target_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __getitem__(self, idx):        
        img_path    = os.path.join(self.root, '', self.imgs[idx])        
        input_img   = Image.open(img_path).convert("RGB")
        target_img  = Image.open(img_path).convert("RGB")

        return self.input_trans(input_img), self.target_trans(target_img)

    def __len__(self):
        return len(self.imgs)