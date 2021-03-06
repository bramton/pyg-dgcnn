import argparse
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch_geometric as tg
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import numpy as np
from utils.extra_tforms import RandomScale, RandomShift, NormalizeArea, Jitter, FixedSize
from model import DGCNN
from ModelNetDGCNN import ModelNetDGCNN

from GeomShapes import GeomShapes

def train(n):
    rdeg = np.degrees(0.18)
    #tf = T.Compose([T.NormalizeScale(),NormalizeArea(target_area=10.24),T.SamplePoints(n),
    #                T.RandomRotate(180, axis=2), Jitter(0.05, loc=0, scale=0.01), 
    #                T.RandomScale((0.9, 1.1)),
    #                T.RandomRotate(rdeg, axis=0),T.RandomRotate(rdeg, axis=1),T.RandomRotate(rdeg, axis=2)])
    #tf = T.Compose([T.NormalizeScale(), T.SamplePoints(n),
    #                #T.RandomRotate(180, axis=2),
    #                Jitter(0.05, loc=0, scale=0.01), 
    #                T.RandomScale((0.9, 1.1)),
    #                T.RandomRotate(rdeg, axis=0),T.RandomRotate(rdeg, axis=1),T.RandomRotate(rdeg, axis=2)])
    tf_geom = T.Compose([T.RandomScale((0.95, 1.05)),
                    T.RandomRotate(180, axis=0),T.RandomRotate(180, axis=1),T.RandomRotate(180, axis=2),
                    RandomShift(0.01)])
    tf = T.Compose([T.SamplePoints(1024), T.NormalizeScale(),RandomScale(2./3.,3./2.), 
                    RandomShift(0.2)])
    tf = T.Compose([T.FixedPoints(1024), RandomScale(2./3.,3./2.), 
                    RandomShift(0.2)])
    #mn = ModelNet("data", name='40', transform=tf)
    mn = ModelNetDGCNN("data/mn_ply", transform=tf)
    #mn = GeomShapes(transform=tf_geom)
    print('Number of samples: {:d}'.format(len(mn)))
    print('Number of classes: {:d}'.format(mn.num_classes))
    LR_START = 0.1
    LR_END = 0.001
    EPOCHS_MAX = 250

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGCNN(mn.num_classes, k=20).to(device).float()
    #optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    optimiser = torch.optim.SGD(model.parameters(), lr=LR_START, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, EPOCHS_MAX, eta_min=LR_END)

    loader = DataLoader(mn, batch_size=32, shuffle=True)
    loader_test = testloader(n=n)

    hist_lr = np.zeros((EPOCHS_MAX,1))
    hist_loss = np.zeros((EPOCHS_MAX,1))
    best_acc = 0

    print("Running training with {:d} points".format(n))
    for epoch in range(EPOCHS_MAX):
        model.train()
        tot_batch = 0
        tot_loss = 0
        tot_seen = 0
        tot_acc = 0

        for batch in loader:
            batch.to(device)

            optimiser.zero_grad()
            raw = model(batch)
            #print(raw.size())
            loss = F.cross_entropy(raw, batch.y, label_smoothing=0.2)
            loss.backward()
            optimiser.step()

            preds = raw.max(dim=1)[1]

            acc = (preds == batch.y).sum()
            tot_acc = tot_acc + acc

            tot_batch = tot_batch + 1
            tot_seen = tot_seen + len(batch.y)
            tot_loss = tot_loss + loss.item()

        scheduler.step()
        hist_loss[epoch] = tot_loss/tot_batch
        hist_lr[epoch] = scheduler.get_last_lr() 
        train_oacc = tot_acc/tot_seen

        # TEST
        tot_seen = 0
        tot_acc = 0

        model.eval()
        with torch.no_grad():
            for batch in loader_test:
                batch.to(device)

                raw = model(batch)
                preds = raw.max(dim=1)[1]

                acc = (preds == batch.y).sum()
                tot_acc = tot_acc + acc

                tot_seen = tot_seen + len(batch.y)

        test_oacc = tot_acc/tot_seen
        if test_oacc > best_acc:
            best_acc = test_oacc

        print('Epoch: {:3d} train loss: {:.2f} acc: {:2f} test acc: {:2f}'.format(epoch,tot_loss/tot_batch, train_oacc, test_oacc))

    print('Best test accuracy: {:2f}'.format(best_acc))
    torch.save(model,'model.pt')

def testloader(n):
    tf = T.Compose([T.FixedPoints(1024)])
    #tf = T.Compose([T.NormalizeScale(), NormalizeArea(target_area=10.24),T.SamplePoints(n)])
    #mn = ModelNet("data", name='40', train=False, transform=tf)
    mn = ModelNetDGCNN("data/mn_ply", train=False, transform=tf)
    loader = DataLoader(mn, batch_size=32, shuffle=True, drop_last=False)
    return loader

def test(n):
    loader = testloader(n=n)
    print('Number of samples: {:d}'.format(len(mn)))
    print('Number of classes: {:d}'.format(mn.num_classes))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGCNN(mn.num_classes, k=20).to(device).float()
    model = torch.load('model.pt')

    model.eval()
    tot_seen = 0
    tot_acc = 0

    for batch in loader:
        batch.to(device)

        raw = model(batch)
        preds = raw.max(dim=1)[1]

        acc = (preds == batch.y).sum()
        tot_acc = tot_acc + acc

        tot_seen = tot_seen + len(batch.y)

    print('acc: {:2f}'.format(tot_acc/tot_seen))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyG Dynamic Graph CNN")
    parser.add_argument('--eval', type=bool, default=False, help='Evaluate model')
    args = parser.parse_args()
    n = 1024
    if args.eval:
        print('eval')
        test(n=n)
    else:
        print('train')
        train(n=n)
