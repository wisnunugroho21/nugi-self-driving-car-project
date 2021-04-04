import torch
from torch.utils.data import DataLoader

from dataloader.CarlaDataset import CarlaDataset
from loss.simclr import SimCLR
from model.main.cnn_model import CnnModel
from model.main.projection_model import ProjectionModel

soft_tau = 0.99
PATH    = '.'
device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainset            = CarlaDataset(root = 'dataset/carla')
trainloader         = DataLoader(trainset, batch_size = 32, shuffle = True, num_workers = 8)

cnn                 = CnnModel().float().to(device)
cnn_target          = CnnModel().float().to(device)

projector           = ProjectionModel().float().to(device)
projector_target    = ProjectionModel().float().to(device)

clroptimizer        = torch.optim.Adam(list(cnn.parameters()) + list(projector.parameters()), lr = 0.001)
clrscaler           = torch.cuda.amp.GradScaler()
clrloss             = SimCLR(True)

cnn_target.load_state_dict(cnn.state_dict())
projector_target.load_state_dict(projector.state_dict())

for epoch in range(500):
    running_loss = 0.0
    for input, target in trainloader:
        input, target   = input.to(device), target.to(device)

        clroptimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out1        = cnn(input)
            encoded1    = projector(out1)

            out2        = cnn_target(target, True)
            encoded2    = projector_target(out2, True)

            loss = clrloss.compute_loss(encoded1, encoded2)

        clrscaler.scale(loss).backward()
        clrscaler.step(clroptimizer)
        clrscaler.update()

    print('loop clr -> ', epoch)

    for target_param, param in zip(cnn_target.parameters(), cnn.parameters()):
        target_param.data.copy_(target_param.data * soft_tau  + param.data  * (1.0 - soft_tau))

    for target_param, param in zip(projector_target.parameters(), projector.parameters()):
        target_param.data.copy_(target_param.data * soft_tau  + param.data  * (1.0 - soft_tau))

print('Finished Pre-Training')
torch.save(cnn.state_dict(), PATH + '/cnn.pth')