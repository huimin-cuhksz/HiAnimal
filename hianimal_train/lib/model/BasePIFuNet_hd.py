import torch.nn as nn

from ..geometry import index, orthogonal


class BasePIFuNet(nn.Module):
    def __init__(self, projection_mode="orthogonal", criteria=None):
        if projection_mode != "orthogonal":
            raise ValueError("This training package only supports orthogonal projection")
        super().__init__()
        self.name = "base"
        self.criteria = criteria or {"occ": nn.MSELoss()}
        self.index = index
        self.projection = orthogonal
        self.preds = None
        self.labels = None

    def get_preds(self):
        return self.preds
