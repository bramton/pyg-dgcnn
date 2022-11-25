# Useful resources:
# https://www.medien.ifi.lmu.de/lehre/ws2122/gp/
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dgcnn_classification.py
#
# Original code:
# https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/models/dgcnn.py

import torch
import torch.nn.functional as F

from torch_geometric.nn import DynamicEdgeConv, MLP, global_max_pool, global_mean_pool
from tNet import tNet
class DGCNN(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr='max'):
        super(DGCNN, self).__init__()
        act_fun = 'leaky_relu'
        act_args = {'negative_slope': 0.2} # Weird, default=0.01
        self.conv1 = DynamicEdgeConv(MLP([2*3,    64], bias=False, plain_last=False, act=act_fun, act_kwargs=act_args), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2*64,   64], bias=False, plain_last=False, act=act_fun, act_kwargs=act_args), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2*64,  128], bias=False, plain_last=False, act=act_fun, act_kwargs=act_args), k, aggr)
        self.conv4 = DynamicEdgeConv(MLP([2*128, 256], bias=False, plain_last=False, act=act_fun, act_kwargs=act_args), k, aggr)

        self.mlp1 = MLP([2*256, 1024], bias=False, plain_last=False, act=act_fun, act_kwargs=act_args)

        bias_list=[False, True, True]
        self.mlp2 = MLP([2*1024, 512, 256, out_channels], bias=bias_list, plain_last=True, act=act_fun, act_kwargs=act_args, dropout=0.5)

    def forward(self, data):
        pos, batch = data.pos.float(), data.batch
        x1 = self.conv1(pos,batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)
        out = self.mlp1(torch.cat([x1, x2, x3, x4], dim=1)) # after cat: 1024x(3*64 + 128)

        x1 = global_max_pool(out, batch)
        x2 = global_mean_pool(out, batch)
        out = torch.cat([x1, x2], dim=1)

        return self.mlp2(out)

class DGCNN_semseg(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr='max'):
        super(DGCNN_semseg, self).__init__()
        act_fun = 'leaky_relu'
        act_args = {'negative_slope': 0.2} # Weird, default=0.01
        self.conv1 = DynamicEdgeConv(MLP([2*9,  64, 64], bias=False, plain_last=False, act=act_fun, act_kwargs=act_args), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2*64, 64, 64], bias=False, plain_last=False, act=act_fun, act_kwargs=act_args), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2*64, 64],     bias=False, plain_last=False, act=act_fun, act_kwargs=act_args), k, aggr)

        self.mlp1 = MLP([3*64, 1024], bias=False, plain_last=False, act=act_fun, act_kwargs=act_args)

        dp_list=[0., 0.7, 0.]
        self.mlp2 = MLP([3*64 + 1024, 512, 256, out_channels], bias=False, plain_last=True, act=act_fun, act_kwargs=act_args, dropout=dp_list)

    def forward(self, data):
        batch = data.batch
        x = torch.cat([data.pos, data.x], dim=1)
        num_points = len(data.y)
        #print(x.size())
        x1 = self.conv1( x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        #print(x3.size())

        out = torch.cat([x1, x2, x3], dim=1) # after cat: 1024x(3*64 + 128)
        #print(out.size())
        out = self.mlp1(out) 
        #print("Na MLP1")
        #print(out.size())
        out = global_max_pool(out, batch)
        #print(out.size())
        out = out.repeat_interleave(4096, dim=0) # Hardcoded! Should be better way based on batch info
        #print(out.size())
        out = torch.cat([out, x1, x2, x3], dim=1)
        out = self.mlp2(out)

        return out

class DGCNN_pseg(torch.nn.Module):
    def __init__(self, out_channels, k=20, part_num=50, aggr='max'):
        super(DGCNN_pseg, self).__init__()
        act_fun = 'leaky_relu'
        act_args = {'negative_slope': 0.2} # Weird, default=0.01

        self.tNet = tNet(k=k)
        self.conv1 = DynamicEdgeConv(MLP([ 2*3, 64, 64], bias=False, plain_last=False, act=act_fun, act_kwargs=act_args), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2*64, 64, 64], bias=False, plain_last=False, act=act_fun, act_kwargs=act_args), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2*64, 64    ], bias=False, plain_last=False, act=act_fun, act_kwargs=act_args), k, aggr)

        self.mlp1 = MLP([3*64, 1024], bias=False, plain_last=False, act=act_fun, act_kwargs=act_args)
        self.mlp2 = MLP([  16, 64],   bias=False, plain_last=False, act=act_fun, act_kwargs=act_args)

        drop_list=[0.6, 0.6, 0, 0]
        self.mlp3 = MLP([1024+64 + 3*64, 256, 256, 128, part_num], plain_last=True, act=act_fun, act_kwargs=act_args, dropout=drop_list)

    def forward(self, data):
        pos, y, batch = data.pos.float(), data.category, data.batch

        # Categorical vector
        y = F.one_hot(y, num_classes=16).float()
        y = self.mlp2(y)

        t = self.tNet(pos, batch)
        pos = torch.matmul(pos.unsqueeze(1), t)
        pos = pos.squeeze()

        x1 = self.conv1(pos,batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.mlp1(torch.cat([x1, x2, x3], dim=1))
        out = global_max_pool(out, batch)

        out = torch.cat([out, y], dim=1)
        out = out.repeat_interleave(2048, dim=0) # Hardcoded! Should be better way based on batch info
        x = torch.cat((out, x1, x2, x3), dim=1)

        return self.mlp3(x)
