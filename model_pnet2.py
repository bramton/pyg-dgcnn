import torch
import torch.nn.functional as F

from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius

class SAModuleMSG(torch.nn.Module):
    def __init__(self, ratio, r, nsample, nn):
        super().__init__()
        self.ratio = ratio # For the Furthest Point sampling
        self.r_list = r # List of radii
        self.nsample = nsample
        self.conv = torch.nn.ModuleList()
        for i in range(len(self.r_list)):
            self.conv.append(PointConv(MLP(nn[i]), add_self_loops=False))

    def forward(self, F, pos, batch):
        #print("*********")
        #print("Feat size: "  + str(F.shape if F is not None else 0))
        #print("pos size: "  + str(pos.shape))
        #print("batch size: "  + str(batch.shape))
        idx = fps(pos, batch, ratio=self.ratio)

        x_list = []
        for i,r in enumerate(self.r_list):
            row, col = radius(pos, pos[idx], r, batch, batch[idx],
                              max_num_neighbors=self.nsample[i])
            edge_index = torch.stack([col, row], dim=0)
            F_dst = None if F is None else F[idx]
            x = self.conv[i]((F,F_dst), (pos, pos[idx]), edge_index)
            #print("na conv size: "  + str(x.shape)) # 512*32 x 128
            #x = global_max_pool(x, batch[idx])
            #print("na max pool size: "  + str(x.shape))
            x_list.append(x)

        out = torch.cat(x_list, dim=1)
        pos, batch = pos[idx], batch[idx]
        #print("out size: "  + str(out.shape))
        return out, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        return x


class PNet2(torch.nn.Module):
    def __init__(self):
        super(PNet2, self).__init__()

        self.sa1 = SAModuleMSG(0.5, [0.1, 0.2, 0.4], [16, 32, 128],
                   [[3, 32, 32, 128],
                   [3, 64, 64, 128],
                   [3, 64, 96, 128]])
        self.sa2 = SAModuleMSG(0.25, [0.2, 0.4, 0.8], [32, 64, 128],
                   [[3*128+3, 64, 64, 128],
                   [3*128+3, 128, 128, 256],
                   [3*128+3, 128, 128, 256]])

        self.sa3 = GlobalSAModule(MLP([128+2*256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, 40], dropout=0.4, plain_last=True)

    def forward(self, data):
        sa0_out = (data.x, data.pos.float(), data.batch)
        sa1_out = self.sa1(*sa0_out)
        sa2_out = self.sa2(*sa1_out)
        x = self.sa3(*sa2_out)

        x = self.mlp(x).log_softmax(dim=-1)

        return x
