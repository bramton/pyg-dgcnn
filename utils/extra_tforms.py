from scipy.stats import truncnorm
from torch_geometric.transforms import BaseTransform
from numpy.random import default_rng
import numpy as np
import torch

class Jitter(BaseTransform):
    def __init__(self, clip, loc=0, scale=1):
        assert(clip > 0)
        self.rv = truncnorm(-clip, clip, loc, scale)
    def __call__(self, data):
        noise = self.rv.rvs(size=tuple(data.pos.shape))
        data.pos = data.pos + noise
        return data
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.clip})'
    
class RandomShift(BaseTransform):
    def __init__(self, shift):
        assert(shift > 0)
        self.rng = default_rng()
        self.shift = shift
    def __call__(self, data):
        rt = self.rng.uniform(-self.shift, self.shift, (1,3))
        data.pos = data.pos + rt
        return data
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.shift})'

class RandomScale(BaseTransform):
    def __init__(self, smin, smax):
        assert(smin > 0)
        assert(smax > 0)
        self.rng = default_rng()
        self.smin = smin
        self.smax = smax
    def __call__(self, data):
        rt = self.rng.uniform(self.smin, self.smax, (1,3))
        data.pos = np.multiply(data.pos, rt)
        return data
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.smin})'

class NormalizeArea(BaseTransform):
    def __init__(self, target_area=1):
        assert(target_area > 0)
        self.ta = target_area
    def __call__(self, data):
        pos = data.pos
        pos_max = pos.abs().max()
        pos = pos / pos_max

        # Thanks to: https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle
        coords = pos.unsqueeze(1)
        faces = data.face.T
        fc = torch.hstack((coords[faces[:,0]],
                           coords[faces[:,1]],
                           coords[faces[:,2]]))
        area = torch.linalg.cross(fc[:,1,:] - fc[:,0,:],
                                  fc[:,2,:] - fc[:,0,:])
        area = 0.5*torch.linalg.vector_norm(area, dim=1)
        scale = torch.sqrt(self.ta/(area.sum()))
        data.pos = pos*scale
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.ta})'
