import torch
from torch.serialization import save
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.transforms import RandomApply

from dataloader.DogBreedDataset import DogBreedDataset
from model.basicclr import BasciClr

device = torch.device("cuda:0")

transform  = transforms.Compose([
    transforms.Resize((256, 256))
])

trans_crop = transforms.Compose([
    transforms.RandomCrop(24),
    transforms.Resize(32)
])

trans_jitter = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    transforms.RandomGrayscale(p=0.2)
])

trainset    = DogBreedDataset('dataset/dog_breed', transforms = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers = 1)

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

import torch.nn as nn
import torch.nn.functional as F

def clrloss(first_encoded, second_encoded):
    encoded         = torch.cat((first_encoded, second_encoded), dim = 0)
    zeros           = torch.zeros(encoded.shape[0]).long().to(device)    
    
    similarity      = torch.nn.functional.cosine_similarity(encoded.unsqueeze(1), encoded.unsqueeze(0), dim = 2)
    return torch.nn.functional.cross_entropy(similarity, zeros)

net = BasciClr()
net.to(device)

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr = 0.001)
scaler    = torch.cuda.amp.GradScaler()

for epoch in range(3):

    running_loss = 0.0
    for data in trainloader:
        inputs, _       = data
        crop_inputs     = trans_crop(inputs).to(device)
        jitter_inputs   = trans_jitter(inputs).to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            _, crop_outputs     = net(crop_inputs)
            _, jitter_outputs   = net(jitter_inputs)

            loss = clrloss(crop_outputs, jitter_outputs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print('loop clr -> ', epoch)

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

del net
del optimizer
del scaler

PATH = './cifar_net.pth'

net = BasciClr()
net.to(device)
net.load_state_dict(torch.load(PATH))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)
scaler    = torch.cuda.amp.GradScaler()

for epoch in range(3):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs, _ = net(inputs)
            loss  = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)