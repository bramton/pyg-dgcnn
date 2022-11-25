from scipy.stats import truncnorm
from torch_geometric.transforms import BaseTransform, Center
from torch_geometric.transforms import LinearTransformation
from numpy.random import default_rng
import numpy as np
import torch
from torch import linalg

class Jitter(BaseTransform):
    def __init__(self, clip, loc=0, scale=1):
        assert(clip > 0)
        a, b = (clip - loc) / scale, (clip - loc) / scale
        self.rv = truncnorm(-a, b, loc, scale)
    def __call__(self, data):
        noise = self.rv.rvs(size=tuple(data.pos.shape))
        data.pos = data.pos + noise
        return data
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.clip})'
    
class RandomRotateArvo(BaseTransform):
    def __init__(self):
        self.rng = default_rng()
    def __call__(self, data):
        x1,x2,x3 = self.rng.uniform(size=3)
        # Random rotation around z
        R = np.eye(3)
        R[0,0] = R[1,1] = np.cos(2*np.pi*x1)
        R[0,1] = np.sin(2*np.pi*x1)
        R[1,0] = -np.sin(2*np.pi*x1)

        v = np.zeros((3,1))
        v[0] = np.cos(2*np.pi*x2)*np.sqrt(x3)
        v[1] = np.sin(2*np.pi*x2)*np.sqrt(x3)
        v[2] = np.sqrt(1.0 - x3)
        H = 2*np.outer(v, v) - np.eye(3)
        M = H @ R
        return LinearTransformation(M)(data)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({x1})'

class FixedSize(BaseTransform):
    def __init__(self, n):
        assert(n > 0)
        self.n = n
    def __call__(self, data):
        data.pos = data.pos[:self.n,:]
        data.y = data.y[:self.n]
        return data
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.n})'

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

class SphereNormalize(BaseTransform):
    def __init__(self):
        self.center = Center()

    def __call__(self, data):
        data = self.center(data)

        scale = (1 / linalg.vector_norm(data.pos, dim=1).max()) * 0.999999
        data.pos = data.pos * scale
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

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
