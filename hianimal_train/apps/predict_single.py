import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from lib.mesh_util import reconstruct, save_obj
from lib.model.HGPIFuNetwNML import HGPIFuNetwNML


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--loadSize", type=int, default=512)
    parser.add_argument("--z_size", type=float, default=200.0)
    parser.add_argument("--num_views", type=int, default=1)
    parser.add_argument("--no_residual", action="store_true")
    parser.add_argument(
        "--mlp_dim", type=int, nargs="+", default=[257, 1024, 512, 256, 128, 3]
    )
    return parser.parse_args()


def main(opt):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HiAnimal prediction")
    device = torch.device(f"cuda:{opt.gpu_id}")

    image = Image.open(opt.input).convert("RGB").resize((512, 512))
    to_tensor = transforms.Compose(
        [
            transforms.Resize(opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    image = to_tensor(image).float().unsqueeze(0).to(device)

    calibration = torch.tensor(
        [
            [0.707, 0.0, -0.707, 0.0],
            [0.0, -1.0, 0.0, 0.8],
            [-0.707, 0.0, -0.707, 3.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    model = HGPIFuNetwNML(opt).to(device)
    model.load_state_dict(torch.load(opt.checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        model.filter(image)
        vertices, faces, uv = reconstruct(
            model,
            device,
            calibration,
            opt.resolution,
            np.array([-1.76, -1.76, -1.76]),
            np.array([1.76, 1.76, 1.76]),
        )
    os.makedirs(os.path.dirname(os.path.abspath(opt.output)), exist_ok=True)
    save_obj(f"{opt.output}.obj", vertices, faces)
    np.savez(f"{opt.output}.npz", uvs=uv)


if __name__ == "__main__":
    main(parse_args())
