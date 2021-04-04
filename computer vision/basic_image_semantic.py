import torch
import torch.nn as nn
import torchvision.transforms as transforms

from dataloader.CatsDataset import CatsDataset
from loss.simclr import SimCLR
from model.image_semantic.encoder import Encoder
from model.image_semantic.decoder import Decoder
from model.clr.projection import Projection

PATH = '.'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trans0 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trans1 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(256),                           
    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p = 0.8),
    transforms.RandomGrayscale(p = 0.2),
    transforms.GaussianBlur(25),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trans2 = transforms.Compose([
    transforms.Resize((256, 256)),    
    transforms.RandomResizedCrop(256),                           
    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p = 0.8),
    transforms.RandomGrayscale(p = 0.2),
    transforms.GaussianBlur(25),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trans_label  = transforms.Compose([
    transforms.Resize((256, 256))
])

trainset    = CatsDataset(root = 'dataset/cats', transforms1 = trans0, transforms2 = trans_label)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 8, shuffle = False, num_workers = 8)

clrset1     = CatsDataset(root = 'dataset/cats', transforms1 = trans1, transforms2 = trans_label)
clrloader1  = torch.utils.data.DataLoader(clrset1, batch_size = 8, shuffle = False, num_workers = 8)

clrset2     = CatsDataset(root = 'dataset/cats', transforms1 = trans2, transforms2 = trans_label)
clrloader2  = torch.utils.data.DataLoader(clrset2, batch_size = 8, shuffle = False, num_workers = 8)

testset     = CatsDataset(root = 'dataset/cats', transforms1 = trans0, transforms2 = trans_label)
testloader  = torch.utils.data.DataLoader(testset, batch_size = 8, shuffle = False, num_workers = 8)

encoder     = Encoder()
projector   = Projection()

encoder, projector = encoder.to(device), projector.to(device)

clroptimizer    = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr = 0.001)
clrscaler       = torch.cuda.amp.GradScaler()
clrloss         = SimCLR(True)

for epoch in range(5):
    running_loss = 0.0
    for data1, data2 in zip(clrloader1, clrloader2):
        input1, _   = data1        
        input2, _   = data2

        input1, input2  = input1.to(device), input2.to(device)

        clroptimizer.zero_grad()

        with torch.cuda.amp.autocast():
            mid1   = encoder(input1).mean([2, 3])
            out1   = projector(mid1)

            mid2   = encoder(input2).mean([2, 3])
            out2   = projector(mid2)

            loss = (clrloss.compute_loss(out1, out2) + clrloss.compute_loss(out2, out1)) / 2.0
        clrscaler.scale(loss).backward()
        clrscaler.step(clroptimizer)
        clrscaler.update()

    print('loop clr -> ', epoch)

print('Finished Pre-Training')
torch.save(encoder.state_dict(), PATH + '/encoder.pth')

decoder = Decoder()
decoder = decoder.to(device)

segoptimizer    = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = 0.001)
segscaler       = torch.cuda.amp.GradScaler()
segloss         = nn.CrossEntropyLoss()

for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        input, label    = data
        input, label    = input.to(device), label.to(device)

        segoptimizer.zero_grad()

        with torch.cuda.amp.autocast():
            mid = encoder(input)
            out = decoder(mid)

            loss = segloss(out, label)

        segscaler.scale(loss).backward()
        segscaler.step(segoptimizer)
        segscaler.update()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

torch.save(encoder.state_dict(), PATH + '/encoder.pth')
torch.save(decoder.state_dict(), PATH + '/decoder.pth')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        mid = encoder(input)
        out = decoder(mid)
        predicted = torch.argmax(out, -1)

        total   += labels.size(0)        
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

torch.save(encoder.state_dict(), PATH + '/encoder.pth')
torch.save(decoder.state_dict(), PATH + '/decoder.pth')
