import torch
import torch.nn as nn

from ..net_util import init_net
from .BasePIFuNet_hd import BasePIFuNet
from .DepthNormalizer import DepthNormalizer
from .HGFilters_hd import HGFilter
from .SurfaceClassifier import SurfaceClassifier


class HGPIFuNetwNML(BasePIFuNet):
    """HiAnimal pixel-aligned occupancy/UV prediction network."""

    def __init__(self, opt, projection_mode="orthogonal", criteria=None):
        super().__init__(
            projection_mode=projection_mode,
            criteria=criteria or {"occ": nn.MSELoss()},
        )
        self.name = "hg_pifu"
        self.image_filter = HGFilter(4, 2, 3, 256, "batch", "ave_pool", False)
        self.mlp = SurfaceClassifier(
            filter_channels=opt.mlp_dim,
            num_views=opt.num_views,
            no_residual=opt.no_residual,
            last_op=nn.Sigmoid(),
        )
        self.spatial_enc = DepthNormalizer(opt)
        self.num_views = opt.num_views
        self.im_feat_list = []
        self.intermediate_preds_list = []
        init_net(self)

    def filter(self, images):
        self.im_feat_list, _, _ = self.image_filter(images)
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def query(self, points, calibrations, transforms=None, labels=None):
        xyz = self.projection(points, calibrations, transforms)
        xy = xyz[:, :2, :]
        self.labels = labels
        spatial_features = self.spatial_enc(xyz, calibs=calibrations)
        self.intermediate_preds_list = [
            self.mlp(torch.cat((self.index(features, xy), spatial_features), dim=1))
            for features in self.im_feat_list
        ]
        self.preds = self.intermediate_preds_list[-1]

    def get_error(self):
        error = sum(
            self.criteria["occ"](prediction, self.labels)
            for prediction in self.intermediate_preds_list
        )
        return error / len(self.intermediate_preds_list)

    def forward(self, images, points, calibrations, labels):
        self.filter(images)
        self.query(points, calibrations, labels=labels)
        return self.get_preds(), self.get_error()
