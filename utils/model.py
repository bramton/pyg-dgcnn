from torch_geometric.nn import DynamicEdgeConv, MLP, global_max_pool
import torch.nn.functional as F

# https://www.medien.ifi.lmu.de/lehre/ws2122/gp/
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dgcnn_classification.py
# Original code: https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/models/dgcnn.py

class DGCNN(torch.nn.Module):
    def __init__(self, out_channels, k=30, aggr='max'):
        super(DGCNN, self).__init__()
        self.conv1 = DynamicEdgeConv(MLP([2*3,  64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2*64,  64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2*64,  64]), k, aggr)
        self.conv4 = DynamicEdgeConv(MLP([2*64, 128]), k, aggr)

        self.lin1 = MLP([3 * 64 + 128, 1024])
        self.mlp = MLP([1024, 512, 256, out_channels], dropout=0.5)

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
        out = self.lin1(torch.cat([x1, x2, x3, x4], dim=1)) # after cat: 1024x(3*64 + 128)
        #print("na cat")
        #print(out.size())
        out = global_max_pool(out, batch)
        #print(out.size())
        out = self.mlp(out)

        #print(out.size())
        #out = F.normalize(out, p=2, dim=-1)
        #print(out.size())
        return out
