import torch.nn as nn


class DepthNormalizer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.load_size = opt.loadSize
        self.z_size = opt.z_size

    def forward(self, xyz, calibs=None, index_feat=None):
        return xyz[:, 2:3, :] * (self.load_size // 2) / self.z_size
