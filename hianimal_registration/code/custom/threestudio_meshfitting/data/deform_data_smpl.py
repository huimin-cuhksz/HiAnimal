import bisect
import copy
import math
import random
from dataclasses import dataclass, field

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_full_projection_matrix,
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *
from .ray_utils import get_ray_directions
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# from .utils import load_and_transform_mesh
import json
import cv2
import copy
import trimesh
import pickle
import numpy as np
from threestudio.utils.mesh import load_obj_mesh  # 确保已添加这行导入

def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
            
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces


def get_w2c_matrix(c2w: Float[Tensor, "B 4 4"]) -> Float[Tensor, "B 4 4"]:
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    return w2c


def load_and_transform_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    print(f"Target mesh vertices: {mesh.vertices.shape[0]}, faces: {mesh.faces.shape[0]}")
    # mesh = trimesh.load(mesh_path, process=False)
    # vertices = np.asarray(mesh.vertices)
    # bb_min, bb_max = np.amin(vertices, axis=0), np.amax(vertices, axis=0)
    # center = (bb_min + bb_max) / 2
    # size = bb_max - bb_min
    # max_size = np.amax(size)
    # vertices = vertices - center  # center at (0,0,0)
    # vertices = vertices / max_size * 2
    # vertices[:, 1] += size[1] / max_size  # y is up, stand on y=0 floor
    # mesh.vertices = copy.deepcopy(vertices)
    return mesh

@dataclass
class DeformDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 256
    width: Any = 256
    org_height: Any = 512
    org_width: Any = 512
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_batch_size: int = 1
    rays_d_normalize: bool = True
    fvoy_deg: float = 39.5978
    normalize_to_center: bool = False

    apply_mask: bool = True
    focal: float = 711.1111 / org_width * width

    src_render_dir: str = None
    src_mesh_dir: str = None
    tgt_render_dir: str = None
    tgt_mesh_dir: str = None
    src_dataset: str = "smpl"
    tgt_dataset: str = "tripo"
    source_meshname: str = None
    target_meshname: str = None
    use_tpose: bool = True
    source_id: str = "3763"
    target_id: str = "7954"
    apply_mask: bool = True
    split_dir: str = None
    deform_dir: str = None
    corr_path: str = "None"

    src2tgt: bool = True
    # subfolder="tpose" if use_tpose else "pose"
    # pair_id: str = None
    # source_id, target_id = pair_id.split("_")


class DeformIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any, split="train") -> None:
        super().__init__()
        self.cfg: DeformDataModuleConfig = cfg
        self.height = self.cfg.height
        self.width = self.cfg.width
        self.split = split

        self.batch_size = self.cfg.batch_size
        self.src_mesh_dir = self.cfg.src_mesh_dir
        self.tgt_mesh_dir = self.cfg.tgt_mesh_dir
        self.source_meshname = self.cfg.source_meshname
        self.target_meshname = self.cfg.target_meshname
        self.src_render_dir = self.cfg.src_render_dir
        self.tgt_render_dir = self.cfg.tgt_render_dir

        self.src_dataset = self.cfg.src_dataset
        self.tgt_dataset = self.cfg.tgt_dataset

        self.deform_dir = self.cfg.deform_dir
        self.source_id = self.cfg.source_id
        self.target_id = self.cfg.target_id
        self.src2tgt = self.cfg.src2tgt
        self.normalize_to_center = self.cfg.normalize_to_center

        self.apply_mask = self.cfg.apply_mask

        self.fvoy = torch.tensor([self.cfg.fvoy_deg * math.pi / 180]).float()

        # with open(os.path.join(self.cfg.split_dir, f"{self.split}.json"), "r") as f:
        #     image_id_list = json.load(f)
        image_id_list = []
        elevation_degs = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
        azimuth_degs = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 
                        180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
        for j in range(len(elevation_degs)):
            for i in range(len(azimuth_degs)):
                image_id = i * len(elevation_degs) + j
                # if elevation_degs[j] not in [-60, -45]:
                image_id_list.append(f"{image_id:04}")

        source_folder = os.path.join(self.src_render_dir, self.source_id)
        target_folder = os.path.join(self.tgt_render_dir, self.target_id)

        # self.src_c2ws = []
        self.tgt_c2ws = []
        self.tgt_depths = []
        self.tgt_normals = []
        # self.src_fg_masks = []
        self.tgt_fg_masks = []
        # self.deform_maps = []
        # self.conf_maps = []
        self.albedos = []
        self.correspondence = None
        self.corr_path = self.cfg.corr_path

        if self.tgt_dataset == "smpl":
            tgt_mesh_path = os.path.join(
                self.tgt_mesh_dir, self.target_id, self.target_meshname
            )
            tgt_mesh = load_and_transform_mesh(tgt_mesh_path)
            tgt_vert = np.asarray(tgt_mesh.vertices)
        elif self.tgt_dataset == "tripo3d":
            tgt_mesh_path = os.path.join(
                self.tgt_mesh_dir, self.target_id, self.target_meshname
            )
            vertices, faces = load_obj_mesh(tgt_mesh_path)  # 使用 load_obj_mesh 替换 trimesh.load
            tgt_vert = vertices

        if self.src_dataset == "smpl":
            src_mesh_path = os.path.join(
                self.src_mesh_dir, self.source_id, self.source_meshname
            )
            src_mesh = load_and_transform_mesh(src_mesh_path)
            src_vert = np.asarray(src_mesh.vertices)
        elif self.src_dataset == "tripo3d":
            src_mesh_path = os.path.join(
                self.src_mesh_dir, self.source_id, self.source_meshname
            )
            vertices, faces = load_obj_mesh(src_mesh_path)  # 使用 load_obj_mesh 替换 trimesh.load
            src_vert = vertices
        """
        vertices is now aligned with the loaded camera pose, however, if the mesh is further scaled, camera pose needs to be updated
        """

        if self.normalize_to_center:
            tgt_vert, tgt_transform = self.normalize_vert_to_center(tgt_vert)
            src_vert, src_transform = self.normalize_vert_to_center(src_vert)
            tgt_transform, src_transform = (
                torch.from_numpy(tgt_transform).float(),
                torch.from_numpy(src_transform).float(),
            )

        self.tgt_vert = torch.from_numpy(tgt_vert).float()
        self.src_vert = torch.from_numpy(src_vert).float()

        for i, image_id in enumerate(image_id_list):
            # src_c2w = self.load_c2w(image_id, source_folder)
            tgt_c2w = self.load_c2w(image_id, target_folder)

            if self.normalize_to_center:
                # src_c2w = torch.matmul(src_transform, src_c2w)
                tgt_c2w = torch.matmul(tgt_transform, tgt_c2w)

            # src_c2w = src_c2w[:3, :4]
            tgt_c2w = tgt_c2w[:3, :4]

            # self.src_c2ws.append(src_c2w)
            self.tgt_c2ws.append(tgt_c2w)
            # self.albedos.append(self.load_albedo(image_id, source_folder))
            self.albedos.append(self.load_albedo(image_id, target_folder))

            # self.src_fg_masks.append(self.load_fg_mask(image_id, source_folder))
            self.tgt_fg_masks.append(self.load_fg_mask(image_id, target_folder))
            self.tgt_normals.append(self.load_normal(image_id, target_folder))
            self.tgt_depths.append(self.load_depth(image_id, target_folder))

        # self.deform_field, self.similarity = self.load_deform_field(
        #     src2tgt=self.src2tgt
        # )
        self.corr_mask, self.deform_field = self.load_correspondence()
        self.dataset_len = len(self.albedos)

    def __iter__(self):
        while True:
            yield {}

    def normalize_vert_to_center(self, vertices):
        vmin = np.amin(vertices, axis=0)
        vmax = np.amax(vertices, axis=0)
        scale = 2 / np.amax(vmax - vmin)
        center = np.mean(vertices, axis=0)
        v_pos = vertices - center  # Center mesh on origin
        v_pos = v_pos * scale

        scale_mat = np.eye(4)
        shift_mat = np.eye(4)
        shift_mat[0:3, 3] = -center
        scale_mat[0, 0], scale_mat[1, 1], scale_mat[2, 2] = scale, scale, scale
        tran_mat = np.dot(scale_mat, shift_mat)
        return v_pos, tran_mat

    def load_fg_mask(self, image_id, render_folder):
        mask_path = os.path.join(render_folder, "mask_" + image_id + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(
            mask, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST
        )
        return torch.from_numpy(mask).float()

    def load_albedo(self, image_id, render_folder):
        albedo_path = os.path.join(render_folder, "albedo_" + image_id + ".jpg")
        albedo = cv2.imread(albedo_path)[:, :, ::-1]
        albedo = cv2.resize(
            albedo, dsize=(self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        albedo = copy.deepcopy(albedo)
        albedo = albedo / 255.0
        return torch.from_numpy(albedo).float()

    def load_depth(self, image_id, render_folder):
        depth_path = os.path.join(render_folder, "depth_" + image_id + ".exr")
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)[
            :, :, 0:1
        ]
        depth = cv2.resize(
            depth, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST
        )
        return torch.from_numpy(depth).float()

    def load_normal(self, image_id, render_folder):
        normal_path = os.path.join(render_folder, "normal_" + image_id + ".exr")
        normal = cv2.imread(normal_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)[
            :, :, ::-1
        ]
        normal = cv2.resize(
            normal, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST
        )
        normal = normal[:, :, [0, 2, 1]]
        normal[:, :, 2] *= -1
        return torch.from_numpy(copy.deepcopy(normal)).float()

    def load_c2w(self, image_id, render_folder):
        c2w_pth = os.path.join(render_folder, "pose_" + image_id + ".txt")
        c2w = np.loadtxt(c2w_pth)
        c2w = torch.from_numpy(c2w)
        # c2w[0:3,3]-=1.0
        c2w = np.dot(
            c2w, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        )
        return torch.from_numpy(c2w).float()
    
    def load_correspondence(self):
        correspondence_path = self.corr_path
        # target to source correspondence
        # shape: target, value: source indices
        correspondence = np.load(correspondence_path)
        correspondence = torch.from_numpy(correspondence)

        # target indices
        indices = torch.where(correspondence >= 0)[0]
        # source indices
        values = correspondence[indices]
        bi_correspondence = torch.stack((indices, values), dim=1)
        print(f"Correspondence data shape: {bi_correspondence.shape}")

        print("tgt_vert shape:", self.tgt_vert.shape)
        print("src_vert shape:", self.src_vert.shape)
        print("bi_correspondence shape:", bi_correspondence.shape)
        print("bi_correspondence max values:", torch.max(bi_correspondence, dim=0)[0])
        print("bi_correspondence min values:", torch.min(bi_correspondence, dim=0)[0])

        deform_field = self.tgt_vert[bi_correspondence[:, 0]] - self.src_vert[bi_correspondence[:, 1]]

        return bi_correspondence, deform_field

    def load_deform_field(self, src2tgt=True):
        deform_field_path = os.path.join(
            self.deform_dir,
            "%s_%s_%s_%s"
            % (self.src_dataset, self.source_id, self.tgt_dataset, self.target_id),
            "deform_field_%s.npz" % ("src2tgt" if src2tgt else "tgt2src"),
        )
        deform_field_data = np.load(deform_field_path)
        deform_key = "%s_deform" % ("src2tgt" if src2tgt else "tgt2src")
        similarity_key = "%s_sim" % ("src2tgt" if src2tgt else "tgt2src")
        print("loading gt deformation field")
        if self.normalize_to_center:
            print("use normalized deformation field")
            deform_key = deform_key + "_norm"
        deform_field = deform_field_data[deform_key]
        similarity = deform_field_data[similarity_key]

        deform_min = np.amin(deform_field)
        deform_max = np.amax(deform_field)
        min_max = np.stack([deform_min, deform_max])
        self.min_max = min_max
        return (
            torch.from_numpy(deform_field).float(),
            torch.from_numpy(similarity).float(),
        )

    def load_deform_map(self, image_id):
        infix = "src2tgt" if self.src2tgt else "tgt2src"
        deform_path = os.path.join(
            self.deform_dir,
            "deform_map_%s_%s" % (self.source_id, self.target_id),
            "deform_map_%s_%s.jpg" % (infix, image_id),
        )
        # print(deform_path, os.path.exists(deform_path))
        deform_map = cv2.imread(deform_path)
        # deform_map = copy.deepcopy(deform_map[:, :, ::-1])
        deform_map = cv2.resize(
            deform_map, dsize=(self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        deform_map = deform_map / 255.0

        conf_path = os.path.join(
            self.deform_dir,
            "deform_map_%s_%s" % (self.source_id, self.target_id),
            "confidence_%s_%s.jpg" % (infix, image_id),
        )
        conf_map = cv2.imread(conf_path)
        conf_map = cv2.resize(
            conf_map, dsize=(self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        conf_map = conf_map / 255.0

        return torch.from_numpy(deform_map).float(), torch.from_numpy(conf_map).float()

    def collate(self, batch) -> Dict[str, Any]:
        index = np.random.randint(0, self.dataset_len)

        # src_c2w = self.src_c2ws[index]
        tgt_c2w = self.tgt_c2ws[index]
        # src_fg_mask = self.src_fg_masks[index]
        tgt_fg_mask = self.tgt_fg_masks[index]
        tgt_depth = self.tgt_depths[index]
        tgt_normal = self.tgt_normals[index]
        albedo = self.albedos[index]

        # src_camera_position = src_c2w[0:3, 3]
        # src_camera_position = src_camera_position.unsqueeze(0)
        tgt_camera_position = tgt_c2w[0:3, 3]
        tgt_camera_position = tgt_camera_position.unsqueeze(0)

        # src_c2w = src_c2w.unsqueeze(0)
        tgt_c2w = tgt_c2w.unsqueeze(0)
        # src_fg_mask = src_fg_mask.unsqueeze(0)
        tgt_fg_mask = tgt_fg_mask.unsqueeze(0)
        tgt_depth = tgt_depth.unsqueeze(0)
        tgt_normal = tgt_normal.unsqueeze(0)
        albedo = albedo.unsqueeze(0)

        proj_mtx = get_projection_matrix(self.fvoy, self.width / self.height, 0.1, 10.0)
        # src_mtx = get_mvp_matrix(src_c2w, proj_mtx)
        tgt_mtx = get_mvp_matrix(tgt_c2w, proj_mtx)
        tgt_w2c = get_w2c_matrix(tgt_c2w)

        # min_max = torch.from_numpy(self.min_max).float().unsqueeze(0)

        return {
            # "deform_map":deform_map,
            # "src_camera_positions": src_camera_position,
            "tgt_camera_positions": tgt_camera_position,
            # "src_c2w": src_c2w,
            # "src_mtx": src_mtx,
            "tgt_c2w": tgt_c2w,
            "tgt_w2c": tgt_w2c,
            "tgt_mtx": tgt_mtx,
            # "conf_map":conf_map,
            # "src_fg_mask": src_fg_mask,
            "tgt_fg_mask": tgt_fg_mask,
            "tgt_depth": tgt_depth,
            "tgt_normal": tgt_normal,
            "albedo": albedo,
            # "min_max": min_max,
            "height": self.height,
            "width": self.width,
            "tgt_vert": self.tgt_vert.unsqueeze(0),
            "deform_field": self.deform_field,
            "deform_field_mask": self.corr_mask,
            # "deform_field": self.deform_field.unsqueeze(0),
            # "similarity": self.similarity.unsqueeze(0),
        }


class DeformDataset(Dataset):
    def __init__(self, cfg: Any, split="val") -> None:
        super().__init__()
        self.cfg: DeformDataModuleConfig = cfg
        self.height = self.cfg.height
        self.width = self.cfg.width
        self.split = split

        self.batch_size = self.cfg.batch_size
        # self.eval_batch_size=self.eval_batch_size
        self.src_mesh_dir = self.cfg.src_mesh_dir
        self.tgt_mesh_dir = self.cfg.tgt_mesh_dir
        self.source_meshname = self.cfg.source_meshname
        self.target_meshname = self.cfg.target_meshname
        self.src_render_dir = self.cfg.src_render_dir
        self.tgt_render_dir = self.cfg.tgt_render_dir
        self.deform_dir = self.cfg.deform_dir
        self.source_id = self.cfg.source_id
        self.target_id = self.cfg.target_id

        self.src_dataset = self.cfg.src_dataset
        self.tgt_dataset = self.cfg.tgt_dataset

        self.src2tgt = self.cfg.src2tgt

        self.apply_mask = self.cfg.apply_mask
        self.normalize_to_center = self.cfg.normalize_to_center

        self.corr_path = self.cfg.corr_path

        self.fvoy = torch.tensor([self.cfg.fvoy_deg * math.pi / 180]).float()

        # with open(os.path.join(self.cfg.split_dir, f"{self.split}.json"), "r") as f:
        #     image_id_list = json.load(f)
        image_id_list = []
        elevation_degs = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
        azimuth_degs = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 
                        180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
        for j in range(len(elevation_degs)):
            for i in range(len(azimuth_degs)):
                image_id = i * len(elevation_degs) + j
                # if elevation_degs[j] not in [-60, -45]:
                image_id_list.append(f"{image_id:04}")

        source_folder = os.path.join(self.src_render_dir, self.source_id)
        target_folder = os.path.join(self.tgt_render_dir, self.target_id)

        self.src_c2ws = []
        self.tgt_c2ws = []
        self.tgt_depths = []
        self.tgt_normals = []
        self.src_fg_masks = []
        self.tgt_fg_masks = []
        self.deform_maps = []
        self.conf_maps = []
        self.albedos = []

        if self.tgt_dataset == "smpl":
            tgt_mesh_path = os.path.join(
                self.tgt_mesh_dir, self.target_id, self.target_meshname
            )
            tgt_mesh = load_and_transform_mesh(tgt_mesh_path)
            tgt_vert = np.asarray(tgt_mesh.vertices)
        elif self.tgt_dataset == "tripo3d":
            tgt_mesh_path = os.path.join(
                self.tgt_mesh_dir, self.target_id, self.target_meshname
            )
            vertices, faces = load_obj_mesh(tgt_mesh_path)  # 使用 load_obj_mesh 替换 trimesh.load
            tgt_vert = vertices

        if self.src_dataset == "smpl":
            src_mesh_path = os.path.join(
                self.src_mesh_dir, self.source_id, self.source_meshname
            )
            src_mesh = load_and_transform_mesh(src_mesh_path)
            src_vert = np.asarray(src_mesh.vertices)
        elif self.src_dataset == "tripo3d":
            src_mesh_path = os.path.join(
                self.src_mesh_dir, self.source_id, self.source_meshname
            )
            vertices, faces = load_obj_mesh(src_mesh_path)  # 使用 load_obj_mesh 替换 trimesh.load
            src_vert = vertices
        """
        vertices is now aligned with the loaded camera pose, however, if the mesh is further scaled, camera pose needs to be updated
        """

        if self.normalize_to_center:
            tgt_vert, tgt_transform = self.normalize_vert_to_center(tgt_vert)
            src_vert, src_transform = self.normalize_vert_to_center(src_vert)
            tgt_transform, src_transform = (
                torch.from_numpy(tgt_transform).float(),
                torch.from_numpy(src_transform).float(),
            )

            # tgt_pt = trimesh.PointCloud(tgt_vert)
            # src_pt = trimesh.PointCloud(src_vert)
            # trimesh.exchange.export.export_mesh(tgt_pt, 'tgt_pt.obj')
            # trimesh.exchange.export.export_mesh(src_pt, 'src_pt.obj')

        self.tgt_vert = torch.from_numpy(tgt_vert).float()
        self.src_vert = torch.from_numpy(src_vert).float()

        for i, image_id in enumerate(image_id_list):
            # src_c2w = self.load_c2w(image_id, source_folder)
            tgt_c2w = self.load_c2w(image_id, target_folder)

            if self.normalize_to_center:
                # src_c2w = torch.matmul(src_transform, src_c2w)
                tgt_c2w = torch.matmul(tgt_transform, tgt_c2w)

            # src_c2w = src_c2w[:3, :4]
            tgt_c2w = tgt_c2w[:3, :4]
            # self.src_c2ws.append(src_c2w)
            self.tgt_c2ws.append(tgt_c2w)
            self.albedos.append(self.load_albedo(image_id, source_folder))

            # self.src_fg_masks.append(self.load_fg_mask(image_id, source_folder))
            self.tgt_fg_masks.append(self.load_fg_mask(image_id, target_folder))
            self.tgt_normals.append(self.load_normal(image_id, target_folder))
            self.tgt_depths.append(self.load_depth(image_id, target_folder))

        # self.deform_field, self.similarity = self.load_deform_field(
        #     src2tgt=self.src2tgt
        # )
        self.corr_mask, self.deform_field = self.load_correspondence()
        # self.dataset_len=2
        self.dataset_len = len(self.albedos)

    def normalize_vert_to_center(self, vertices):
        vmin = np.amin(vertices, axis=0)
        vmax = np.amax(vertices, axis=0)
        scale = 2 / np.amax(vmax - vmin)
        center = np.mean(vertices, axis=0)
        v_pos = vertices - center  # Center mesh on origin
        v_pos = v_pos * scale

        scale_mat = np.eye(4)
        shift_mat = np.eye(4)
        shift_mat[0:3, 3] = -center
        scale_mat[0, 0], scale_mat[1, 1], scale_mat[2, 2] = scale, scale, scale
        tran_mat = np.dot(scale_mat, shift_mat)
        return v_pos, tran_mat

    def load_fg_mask(self, image_id, render_folder):
        # mask_path = os.path.join(render_folder, "mask_" + image_id + ".exr")
        # mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)[
        #     :, :, 0:1
        # ]
        # mask = cv2.resize(
        #     mask, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST
        # )
        mask_path = os.path.join(render_folder, "mask_" + image_id + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(
            mask, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST
        )
        mask[mask > 0] = 1.0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=1)

        return torch.from_numpy(mask).float()

    def load_albedo(self, image_id, render_folder):
        albedo_path = os.path.join(render_folder, "albedo_" + image_id + ".jpg")
        albedo = cv2.imread(albedo_path)[:, :, ::-1]
        albedo = cv2.resize(
            albedo, dsize=(self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        albedo = copy.deepcopy(albedo)
        albedo = albedo / 255.0
        return torch.from_numpy(albedo).float()

    def load_depth(self, image_id, render_folder):
        depth_path = os.path.join(render_folder, "depth_" + image_id + ".exr")
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)[
            :, :, 0:1
        ]
        depth = cv2.resize(
            depth, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST
        )
        return torch.from_numpy(depth).float()

    def load_normal(self, image_id, render_folder):
        normal_path = os.path.join(render_folder, "normal_" + image_id + ".exr")
        normal = cv2.imread(normal_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)[
            :, :, ::-1
        ]
        normal = cv2.resize(
            normal, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST
        )
        normal = normal[:, :, [0, 2, 1]]
        normal[:, :, 2] *= -1
        return torch.from_numpy(copy.deepcopy(normal)).float()

    def load_c2w(self, image_id, render_folder):
        c2w_pth = os.path.join(render_folder, "pose_" + image_id + ".txt")
        c2w = np.loadtxt(c2w_pth)
        c2w = torch.from_numpy(c2w[:4, :4])
        # c2w[0:3,3]-=1.0
        c2w = np.dot(
            c2w, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        )
        return torch.from_numpy(c2w).float()
    
    def load_correspondence(self):
        correspondence_path = self.corr_path
        # target to source correspondence
        # shape: target, value: source indices
        correspondence = np.load(correspondence_path)
        correspondence = torch.from_numpy(correspondence)

        # target indices
        indices = torch.where(correspondence >= 0)[0]
        # source indices
        values = correspondence[indices]
        bi_correspondence = torch.stack((indices, values), dim=1)

        # deform_field = self.tgt_vert - self.src_vert[correspondence, :]
        deform_field = self.tgt_vert[bi_correspondence[:, 0]] - self.src_vert[bi_correspondence[:, 1]]

        return bi_correspondence, deform_field

    def load_deform_map(self, image_id):
        infix = "src2tgt" if self.src2tgt else "tgt2src"
        deform_path = os.path.join(
            self.deform_dir,
            "deform_map_%s_%s" % (self.source_id, self.target_id),
            "deform_map_%s_%s.jpg" % (infix, image_id),
        )
        # print(deform_path, os.path.exists(deform_path))
        deform_map = cv2.imread(deform_path)
        # deform_map = copy.deepcopy(deform_map[:, :, ::-1])
        deform_map = cv2.resize(
            deform_map, dsize=(self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        deform_map = deform_map / 255.0

        conf_path = os.path.join(
            self.deform_dir,
            "deform_map_%s_%s" % (self.source_id, self.target_id),
            "confidence_%s_%s.jpg" % (infix, image_id),
        )
        conf_map = cv2.imread(conf_path)
        conf_map = cv2.resize(
            conf_map, dsize=(self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        conf_map = conf_map / 255.0

        return torch.from_numpy(deform_map).float(), torch.from_numpy(conf_map).float()

    def load_deform_field(self, src2tgt=True):
        deform_field_path = os.path.join(
            self.deform_dir,
            "%s_%s_%s_%s"
            % (self.src_dataset, self.source_id, self.tgt_dataset, self.target_id),
            "deform_field_%s.npz" % ("src2tgt" if src2tgt else "tgt2src"),
        )
        deform_field_data = np.load(deform_field_path)
        deform_key = "%s_deform" % ("src2tgt" if src2tgt else "tgt2src")
        similarity_key = "%s_sim" % ("src2tgt" if src2tgt else "tgt2src")
        if self.normalize_to_center:
            deform_key = deform_key + "_norm"
        deform_field = deform_field_data[deform_key]
        similarity = deform_field_data[similarity_key]

        deform_min = np.amin(deform_field)
        deform_max = np.amax(deform_field)
        min_max = np.stack([deform_min, deform_max])
        self.min_max = min_max
        return (
            torch.from_numpy(deform_field).float(),
            torch.from_numpy(similarity).float(),
        )

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        # src_c2w = self.src_c2ws[index]
        tgt_c2w = self.tgt_c2ws[index]
        # deform_map = self.deform_maps[index]
        # map_mask = self.map_masks[index]
        # conf_map = self.conf_maps[index]
        # src_fg_mask = self.src_fg_masks[index]
        tgt_fg_mask = self.tgt_fg_masks[index]
        tgt_depth = self.tgt_depths[index]
        tgt_normal = self.tgt_normals[index]
        albedo = self.albedos[index]

        # src_camera_position = src_c2w[0:3, 3]
        tgt_camera_position = tgt_c2w[0:3, 3]

        proj_mtx = get_projection_matrix(
            self.fvoy, self.width / self.height, 0.01, 100.0
        )
        # src_mtx = get_mvp_matrix(src_c2w.unsqueeze(0), proj_mtx)
        tgt_mtx = get_mvp_matrix(tgt_c2w.unsqueeze(0), proj_mtx)
        tgt_w2c = get_w2c_matrix(tgt_c2w.unsqueeze(0)).squeeze(0)

        # min_max = torch.from_numpy(self.min_max).float()

        return {
            # "deform_map": deform_map,
            # "src_camera_positions": src_camera_position,
            "tgt_camera_positions": tgt_camera_position,
            # "src_c2w": src_c2w,
            # "src_mtx": src_mtx.squeeze(0),
            "tgt_c2w": tgt_c2w,
            "tgt_w2c": tgt_w2c,
            "tgt_mtx": tgt_mtx.squeeze(0),
            # "conf_map": conf_map,
            # "src_fg_mask": src_fg_mask,
            "tgt_fg_mask": tgt_fg_mask,
            "tgt_depth": tgt_depth,
            "tgt_normal": tgt_normal,
            "albedo": albedo,
            # "min_max": min_max,
            "index": index,
            "tgt_vert": self.tgt_vert,
            "deform_field": self.deform_field,
            # "similarity": self.similarity,
            "deform_field_mask": self.corr_mask,
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.height, "width": self.cfg.width})
        return batch


@register("deform-data-datamodule-smpl")
class DeformDataModule(pl.LightningDataModule):
    cfg: DeformDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        # print(DeformDataModuleConfig.normalize_to_center,cfg)
        # print(DeformDat)
        # print(cfg)
        # DeformDataModuleConfig(**cfg)
        self.cfg = parse_structured(DeformDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = DeformIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = DeformDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = DeformDataset(self.cfg, "val")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )


