import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL.ImageFilter import GaussianBlur
from torch.utils.data import Dataset


class TrainDataset_Refine(Dataset):
    """Normal-map inputs and occupancy/UV point labels used by HiAnimal."""

    def __init__(self, opt, phase="train"):
        self.opt = opt
        self.root = opt.dataroot
        self.projection_mode = "orthogonal"
        self.is_train = phase == "train"
        self.num_sample_inout = opt.num_sample_inout
        self.samples = self._get_samples()

        self.to_tensor = transforms.Compose(
            [
                transforms.Resize(opt.loadSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.augmentation = transforms.ColorJitter(
            brightness=opt.aug_bri,
            contrast=opt.aug_con,
            saturation=opt.aug_sat,
            hue=opt.aug_hue,
        )

    def _get_samples(self):
        samples = []
        for directory in os.listdir(self.root):
            if "train+" not in directory:
                continue
            normal_root = os.path.join(self.root, directory, "sn_normal")
            for filename in os.listdir(normal_root):
                if filename.endswith(".png"):
                    samples.append((directory, os.path.splitext(filename)[0]))

        random.shuffle(samples)
        return samples

    def _get_image_info(self, subject):
        calibration_path = os.path.join(
            self.root, subject[0], "calib", f"{subject[1]}.txt"
        )
        calibration_data = np.loadtxt(calibration_path, dtype=float)
        calibration = torch.from_numpy(
            calibration_data[4:8, :4] @ calibration_data[:4, :4]
        ).float()

        normal_path = os.path.join(
            self.root, subject[0], "sn_normal", f"{subject[1]}.png"
        )
        normal = Image.open(normal_path).convert("RGB").resize((512, 512))
        if self.is_train:
            normal = self.augmentation(normal)
            if self.opt.aug_blur > 1e-5:
                normal = normal.filter(GaussianBlur(np.random.uniform(0, self.opt.aug_blur)))

        return {
            "img": self.to_tensor(normal).float().unsqueeze(0),
            "calib": calibration.unsqueeze(0),
        }

    @staticmethod
    def _remove_nan_uv(samples, uv_labels):
        valid = ~torch.isnan(uv_labels).any(dim=1)
        return samples[valid], uv_labels[valid]

    def _subsample(self, samples, count, occupancy, uv_labels):
        samples, uv_labels = self._remove_nan_uv(samples, uv_labels)
        indices = (torch.rand(count) * samples.shape[0]).long()
        occupancy_labels = torch.full((count, 1), float(occupancy))
        return samples[indices], torch.cat((occupancy_labels, uv_labels[indices]), dim=1)

    def _sample_points(self, subject):
        sample_root = self.root.replace(
            "36views", "36views_sample_uv_whole"
        )
        sample_path = os.path.join(sample_root, subject[0], "geo", "scale_uv.npz")
        sample_data = np.load(sample_path, allow_pickle=True)

        strategy = (
            ("surface_inside", self.num_sample_inout * 4, 1),
            ("surface_outside", self.num_sample_inout * 4, 0),
            ("random_inside", self.num_sample_inout // 4, 1),
            ("random_outside", self.num_sample_inout // 4, 0),
        )
        samples = []
        labels = []
        for key, count, occupancy in strategy:
            group = sample_data[key].item()
            selected_samples, selected_labels = self._subsample(
                torch.from_numpy(group["points"]),
                count,
                occupancy,
                torch.from_numpy(group["uv_coords"]),
            )
            samples.append(selected_samples)
            labels.append(selected_labels)

        samples = torch.cat(samples, dim=0)
        labels = torch.cat(labels, dim=0)
        indices = (torch.rand(self.num_sample_inout) * samples.shape[0]).long()
        return {
            "samples": samples[indices].float().T,
            "labels": labels[indices].float().T,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        subject = self.samples[index]
        item = {"name": subject}
        item.update(self._get_image_info(subject))
        item.update(self._sample_points(subject))
        return item
