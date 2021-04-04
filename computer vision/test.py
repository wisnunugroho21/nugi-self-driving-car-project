import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from dataloader.PennFudanPedDataset import PennFudanPedDataset
from model.deeplabv4 import Deeplabv4

device  = torch.device('cuda:0')
PATH    = 'weights/deeplabv4_net.pth'

def display(display_list, title):
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])

        disImg  = display_list[i].detach().numpy()
        plt.imshow(disImg)
        plt.axis('off')
    plt.show()

transform1 = transforms.Compose([
    transforms.Resize((256, 256))
])

transform2 = transforms.Compose([
    transforms.Resize((256, 256))
])

dataset     = PennFudanPedDataset('dataset/PennFudanPed', transform1, transform2)
trainloader = data.DataLoader(dataset, batch_size = 32, shuffle = True, num_workers = 1)

net         = Deeplabv4(num_classes = 3).to(device)
net.load_state_dict(torch.load(PATH))
net.eval()

inputs, labels = dataset[20]
inputs  = inputs.unsqueeze(0).to(device)
labels = labels.to(device)

outputs = net(inputs)

disInput    = inputs.squeeze(0).transpose(0, 1).transpose(1, 2)
disOutput   = nn.functional.softmax(outputs[0], 0).argmax(0)

display([disInput.cpu(), labels.cpu(), disOutput.cpu()], ['Input Image', 'True Mask', 'Predicted Mask'])

truth       = (disOutput == labels).sum() / (labels.shape[0] * labels.shape[1]) * 100
print(truth)
