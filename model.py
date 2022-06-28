# Useful resources:
# https://www.medien.ifi.lmu.de/lehre/ws2122/gp/
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dgcnn_classification.py
#
# Original code:
# https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/models/dgcnn.py

import torch
from torch_geometric.nn import DynamicEdgeConv, MLP, global_max_pool, global_mean_pool

class DGCNN(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr='max'):
        super(DGCNN, self).__init__()
        act_fun = 'leaky_relu'
        act_args = {'negative_slope': 0.2} # Weird, default=0.01
        self.conv1 = DynamicEdgeConv(MLP([2*3,   64], act=act_fun, act_kwargs=act_args), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2*64,  64], act=act_fun, act_kwargs=act_args), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2*64,  128], act=act_fun, act_kwargs=act_args), k, aggr)
        self.conv4 = DynamicEdgeConv(MLP([2*128, 256],act=act_fun, act_kwargs=act_args), k, aggr)
        
        self.mlp1 = MLP([2*256, 1024], act=act_fun, act_kwargs=act_args)
        self.mlp2 = MLP([2*1024, 512, 256, out_channels], act=act_fun, act_kwargs=act_args, dropout=0.5)
    
    def forward(self, data):
        pos, batch = data.pos.float(), data.batch
        #print(batch)
        #print(pos.size())
        x1 = self.conv1(pos,batch)
        #print(x1.size())
        x2 = self.conv2(x1, batch)
        #print(x2.size())
        x3 = self.conv3(x2, batch)
        #print(x3.size())
        x4 = self.conv4(x3, batch)
        #print(x4.size())
        out = self.mlp1(torch.cat([x1, x2, x3, x4], dim=1)) # after cat: 1024x(3*64 + 128)
        #print("na cat")
        #print(out.size())
        x1 = global_max_pool(out, batch)
        x2 = global_mean_pool(out, batch)
        out = torch.cat([x1, x2], dim=1)
        #print(out.size())
        out = self.mlp2(out)

        #print(out.size())
        #out = F.normalize(out, p=2, dim=-1)
        #print(out.size())
        return out
