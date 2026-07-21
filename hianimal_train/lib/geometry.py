import torch
import torch.nn.functional as F


def index(features, uv):
    uv = uv.transpose(1, 2).unsqueeze(2)
    batch, points = uv.shape[:2]
    channels = features.shape[1]
    samples = F.grid_sample(features, uv, align_corners=True)
    return samples.view(batch, channels, points)


def orthogonal(points, calibrations, transforms=None):
    rotation = calibrations[:, :3, :3]
    translation = calibrations[:, :3, 3:4]
    projected = torch.baddbmm(translation, rotation, points)
    if transforms is not None:
        scale = transforms[:, :2, :2]
        shift = transforms[:, :2, 2:3]
        projected[:, :2, :] = torch.baddbmm(
            shift, scale, projected[:, :2, :]
        )
    return projected
