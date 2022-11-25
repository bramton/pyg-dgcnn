import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from torch_cluster import knn
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from torch_geometric.nn import MLP, global_max_pool, max_pool_x


# Credits to: An Tao
# https://github.com/antao97/dgcnn.pytorch/blob/master/model.py
class tNet(MessagePassing):
    def __init__(self, k=20):
        super().__init__(aggr=None)
        self.k = k
        act_fun = 'leaky_relu'
        act_args = {'negative_slope': 0.2} # Weird, default=0.01
        self.mlp1 = MLP([6, 64, 128], bias=False, plain_last=False, act=act_fun, act_kwargs=act_args)
        self.mlp2 = MLP([128, 1024], bias=False, plain_last=False, act=act_fun, act_kwargs=act_args)

        self.lin1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.lin2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x, batch):
        edge_index = knn(x, x, self.k, batch, batch)
        # First row are clusters, second row neighbours
        return self.propagate(edge_index, x=x, size=None, batch=batch)

    def message(self, edge_index, x_i: Tensor, x_j: Tensor, batch) -> Tensor:
        edge_feats = torch.cat([x_i, x_j - x_i], dim=-1)
        x = self.mlp1(edge_feats)
        x, index = max_pool_x(edge_index[0,:], x, batch.index_select(0, edge_index[0,:]))
        x = self.mlp2(x)
        x = global_max_pool(x, batch)

        x = F.leaky_relu(self.bn1(self.lin1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn2(self.lin2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)
        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(-1, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x.index_select(0, batch)

    def aggregate(self,x):
        return x


class pseg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tnet = tNet2()

    def forward(self, data):
        pos, batch = data.pos.float(), data.batch
        a = self.tnet(pos, batch)

if __name__ == "__main__":
    n = 1024
    mn = ModelNet('data/mn_2048', name='40', train=False, transform=T.SamplePoints(n))
    loader = DataLoader(mn, batch_size=32, shuffle=True, drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = pseg()#.to(device).float()
    for batch in loader:
        model(batch)
        break
