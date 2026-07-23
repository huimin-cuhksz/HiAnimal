from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import trimesh
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, IterableDataset

from threestudio import register
from threestudio.utils.config import parse_structured


@dataclass
class GeometryRegistrationDataConfig:
    source_mesh: str = "../template/registration_template.obj"
    target_mesh: str = "../../hianimal_train/test_results/cat/cat_prediction.obj"
    corr_path: str = "../inputs/cat/correspondence.npy"
    normalize_to_center: bool = True


def _load_vertices(path: str) -> np.ndarray:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return np.asarray(mesh.vertices)


def _normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    scale = 2.0 / np.amax(np.amax(vertices, axis=0) - np.amin(vertices, axis=0))
    return (vertices - np.mean(vertices, axis=0)) * scale


def _load_registration(cfg: GeometryRegistrationDataConfig) -> Dict[str, torch.Tensor]:
    source = _load_vertices(cfg.source_mesh)
    target = _load_vertices(cfg.target_mesh)
    if cfg.normalize_to_center:
        source = _normalize_vertices(source)
        target = _normalize_vertices(target)

    correspondence = np.load(cfg.corr_path)
    if correspondence.ndim != 1 or correspondence.shape[0] != target.shape[0]:
        raise ValueError(
            "correspondence.npy must contain one source index per target vertex"
        )

    target_indices = np.flatnonzero(correspondence >= 0)
    source_indices = correspondence[target_indices].astype(np.int64)
    if source_indices.size == 0 or source_indices.max() >= source.shape[0]:
        raise ValueError("correspondence.npy contains invalid source indices")

    source_tensor = torch.from_numpy(source).float()
    target_tensor = torch.from_numpy(target).float()
    pairs = torch.from_numpy(
        np.stack((target_indices, source_indices), axis=1)
    ).long()
    deform_field = target_tensor[pairs[:, 0]] - source_tensor[pairs[:, 1]]

    print(
        f"Registration data: source={source.shape[0]} vertices, "
        f"target={target.shape[0]} vertices, correspondences={pairs.shape[0]}"
    )
    return {
        "tgt_vert": target_tensor,
        "deform_field": deform_field,
        "deform_field_mask": pairs,
    }


class GeometryRegistrationIterableDataset(IterableDataset):
    def __init__(self, cfg: GeometryRegistrationDataConfig) -> None:
        super().__init__()
        self.data = _load_registration(cfg)

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        return {
            "tgt_vert": self.data["tgt_vert"].unsqueeze(0),
            "deform_field": self.data["deform_field"],
            "deform_field_mask": self.data["deform_field_mask"],
        }


class GeometryRegistrationDataset(Dataset):
    def __init__(self, cfg: GeometryRegistrationDataConfig) -> None:
        super().__init__()
        self.data = _load_registration(cfg)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {**self.data, "index": torch.tensor(index)}


@register("geometry-registration-datamodule")
class GeometryRegistrationDataModule(pl.LightningDataModule):
    cfg: GeometryRegistrationDataConfig

    def __init__(
        self, cfg: Optional[Union[dict, DictConfig]] = None
    ) -> None:
        super().__init__()
        self.cfg = parse_structured(GeometryRegistrationDataConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = GeometryRegistrationIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = GeometryRegistrationDataset(self.cfg)
        if stage in [None, "test", "predict"]:
            self.test_dataset = GeometryRegistrationDataset(self.cfg)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=0,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=1, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=1, num_workers=0)
