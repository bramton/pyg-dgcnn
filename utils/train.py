import torch
import torch_geometric as tg
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import numpy as np
from utils.random_tforms import Jitter, RandomShift
from model import DGCNN

rdeg = np.degrees(0.18)
tf = T.Compose([T.SamplePoints(1024), T.NormalizeScale(), 
                T.RandomRotate(180, axis=2), Jitter(0.05, loc=0, scale=0.01), 
                T.RandomScale((0.8, 1.25)),
                T.RandomRotate(rdeg, axis=0),T.RandomRotate(rdeg, axis=1),T.RandomRotate(rdeg, axis=2),
                RandomShift(0.1)])
mn = ModelNet("data", name='40', transform=tf)
print('Number of samples: {:d}'.format(len(mn)))
print('Number of classes: {:d}'.format(mn.num_classes))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGCNN(mn.num_classes, k=30).to(device).float()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

loader = DataLoader(mn, batch_size=32, shuffle=True)

model.train()
tot_batch = 0
tot_loss = 0
tot_seen = 0
tot_acc = 0
for batch in loader:
    batch.to(device)

    optimizer.zero_grad()
    raw = model(batch)
    print(raw.size())
    loss = F.cross_entropy(raw, batch.y)
    loss.backward()
    optimiser.step()

    acc = 

    tot_batch = tot_batch + 1
    tot_seen = tot_seen + len(batch.y)
    tot_loss = tot_loss + loss
    print('Train loss: {:.2f}'.format(tot_loss/tot_batch))
    break
