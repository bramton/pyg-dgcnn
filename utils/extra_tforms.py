from scipy.stats import truncnorm
from torch_geometric.transforms import BaseTransform
from numpy.random import default_rng

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