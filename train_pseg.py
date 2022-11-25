import argparse
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch_geometric as tg
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import numpy as np
from utils.extra_tforms import RandomScale, RandomShift, SphereNormalize
from model import DGCNN_pseg

def train(dataset, k, n, uid):
    LR_START = 0.1
    LR_END = 0.001
    EPOCHS_MAX = 250

    tf = T.Compose([T.FixedPoints(n), RandomScale(2./3.,3./2.), RandomShift(0.2)])
    mn = ShapeNet(dataset, include_normals=False, transform=tf)
    print('Number of samples: {:d}'.format(len(mn)))
    print('Number of classes: {:d}'.format(mn.num_classes))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGCNN_pseg(mn.num_classes, k=k).to(device).float()
    optimiser = torch.optim.SGD(model.parameters(), lr=LR_START, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, EPOCHS_MAX, eta_min=LR_END)

    loader = DataLoader(mn, batch_size=32, shuffle=True)
    loader_test = testloader(dataset, n)

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
            # Only start saving the best model, when approaching the end of the training session.
            if epoch > 180:
                torch.save(model, 'model_best_{:s}.pt'.format(uid))

        print('Epoch: {:3d} train loss: {:.2f} acc: {:2f} test acc: {:2f}'.format(epoch,tot_loss/tot_batch, train_oacc, test_oacc))

    print('Best test accuracy: {:2f}'.format(best_acc))
    torch.save(model, 'model_last_{:s}.pt'.format(uid))

def testloader(dataset, n):
    mn = ShapeNet(dataset, include_normals=False, transform=T.FixedPoints(n))
    loader = DataLoader(mn, batch_size=32, shuffle=True, drop_last=False)
    return loader

def test(dataset, k, n, uid):
    loader = testloader(dataset, n)
    mn = loader.dataset
    print('Number of samples: {:d}'.format(len(mn)))
    print('Number of classes: {:d}'.format(mn.num_classes))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGCNN(mn.num_classes, k=k).to(device).float()
    model = torch.load('model_best_{:s}.pt'.format(uid))

    model.eval()
    tot_seen = 0
    tot_acc = 0
    class_seen = np.zeros((mn.num_classes))
    class_acc = np.zeros((mn.num_classes))

    with torch.no_grad():
        for batch in loader:
            batch.to(device)

            raw = model(batch)
            preds = raw.max(dim=1)[1]
            preds = (preds == batch.y).cpu()
            
            # Per class
            for cls in range(mn.num_classes):
                class_seen[cls] = class_seen[cls] + (cls == batch.y.cpu()).sum()
                class_acc[cls] = class_acc[cls] + preds[cls == batch.y.cpu()].sum()

            acc = preds.sum()
            tot_acc = tot_acc + acc

            tot_seen = tot_seen + len(batch.y)

    print(class_seen)
    #print(class_acc)
    #print(class_acc/class_seen)
    print('overall accuracy: {:2f}'.format(tot_acc/tot_seen))
    print('mean class accuracy: {:2f}'.format(np.nan_to_num((class_acc/class_seen)).mean()))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyG Dynamic Graph CNN")
    parser.add_argument('--eval', type=bool, default=False, help='Evaluate model')
    parser.add_argument('-n', type=int, default=1024, help='Number of points')
    parser.add_argument('-k', type=int, default=20, help='Number of nearest neighbours')
    parser.add_argument('-dataset', type=str, default='data/mn', help='Dataset')
    parser.add_argument('-uid', type=str, default='test', help='Unique identifier of this experiment')

    args = parser.parse_args()
    n = args.n
    k = args.k
    dataset = args.dataset
    print(dataset)
    uid = args.uid
    print("using dataset: {:s}".format(dataset))

    if args.eval:
        print('eval')
        test(dataset, k, n, uid)
    else:
        print('train')
        train(dataset, k, n, uid)
