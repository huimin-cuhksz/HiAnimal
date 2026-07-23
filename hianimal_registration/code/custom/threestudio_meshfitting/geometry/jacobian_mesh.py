import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import (
    BaseExplicitGeometry,
    BaseGeometry,
    contract_to_unisphere,
)
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import scale_tensor
from threestudio.utils.typing import *
import trimesh
import copy
import pdb
import pickle
from NeuralJacobianFields.SourceMesh import SourceMesh

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

def read_obj_vertices(file_path):
    vertices = []
    faces = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                if line.startswith("v "):
                    parts = line.strip().split()
                    vertex = [float(coord) for coord in parts[1:]]
                    vertices.append(vertex)
                elif line.startswith("f "):
                    parts = line.strip().split()
                    face = []
                    for part in parts[1:]:
                        vertex_index = int(part.split("/")[0]) - 1
                        face.append(vertex_index)
                    faces.append(face)
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"读取文件时出错: {e}")

    return np.asarray(vertices), np.asarray(faces)

'''
def load_smpl(mesh_path):
    # Now read the smpl model.
    with open(mesh_path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        data = u.load()
        # data = pickle.load(f, encoding='iso-8859-1')
        Vertices = data["v_template"]  ##  Loaded vertices of size (6890, 3)
        faces = data['f']
    return Vertices, faces
'''
def load_smpl(mesh_path):
    #mesh = trimesh.load(mesh_path, process=False)
    #vertices = mesh.vertices
    #faces = mesh.faces
    vertices, faces = read_obj_vertices(mesh_path)
    return vertices, faces


def load_and_transform_mesh(mesh_path):
    # mesh=trimesh.load(mesh_path)
    # vertices, faces = read_obj_vertices(mesh_path)
    vertices, faces = load_smpl(mesh_path=mesh_path)
    print(f"Source mesh vertices: {vertices.shape[0]}, faces: {faces.shape[0]}")
    print(vertices.shape, faces.shape)
    bb_min, bb_max = np.amin(vertices, axis=0), np.amax(vertices, axis=0)
    center = (bb_min + bb_max) / 2
    size = bb_max - bb_min
    max_size = np.amax(size)
    vertices = vertices - center  # center at (0,0,0)
    vertices = vertices / max_size * 2
    vertices[:, 1] += size[1] / max_size  # y is up, stand on y=0 floor
    #mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # mesh.vertices=copy.deepcopy(vertices)
    return mesh


def normalize_vert_to_center(vertices):
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


@threestudio.register("jacobian-mesh")
class CustomMesh(BaseExplicitGeometry):
    @dataclass
    class Config(BaseExplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        shape_init: str = ""
        shape_init_params: Optional[Any] = None
        shape_init_mesh_up: str = "+z"
        shape_init_mesh_front: str = "+x"
        normalize_mesh: bool = False
        normalize_to_center: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        )
        self.feature_network = get_mlp(
            self.encoding.n_output_dims,
            self.cfg.n_feature_dims,
            self.cfg.mlp_network_config,
        )

        # Initialize custom mesh
        if self.cfg.shape_init.startswith("mesh:"):
            assert isinstance(self.cfg.shape_init_params, float)
            mesh_path = self.cfg.shape_init[5:]
            if not os.path.exists(mesh_path):
                raise ValueError(f"Mesh file {mesh_path} does not exist.")

            if self.cfg.normalize_mesh:
                mesh = load_and_transform_mesh(mesh_path)
                # mesh=trimesh.load(mesh_path)
            else:
                scene = trimesh.load(mesh_path)
                if isinstance(scene, trimesh.Trimesh):
                    mesh = scene
                elif isinstance(scene, trimesh.scene.Scene):
                    mesh = trimesh.Trimesh()
                    for obj in scene.geometry.values():
                        mesh = trimesh.util.concatenate([mesh, obj])
                else:
                    raise ValueError(f"Unknown mesh type at {mesh_path}.")
            if self.cfg.normalize_to_center:
                vertices = np.asarray(mesh.vertices)
                new_vertices, _ = normalize_vert_to_center(vertices)
                mesh.vertices = new_vertices

            v_pos = torch.tensor(mesh.vertices, dtype=torch.float32).to(self.device)
            t_pos_idx = torch.tensor(mesh.faces, dtype=torch.int64).to(self.device)
            self.mesh = Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)
            tmp_dir = os.path.join(os.path.dirname(mesh_path), "tmp")
            if os.path.exists(tmp_dir):
                os.system("rm -r %s" % (tmp_dir))
            os.makedirs(tmp_dir, exist_ok=True)
            self.src_jacobian = SourceMesh(0, tmp_dir, {}, 1, ttype=torch.float)
            self.src_jacobian.load(source_v=mesh.vertices, source_f=mesh.faces)
            self.src_jacobian.to(self.device)
            jacobian_init = self.src_jacobian.jacobians_from_vertices(
                v_pos.unsqueeze(0)
            ).squeeze(0)
            jacobian_init = jacobian_init.reshape(jacobian_init.shape[0], 9)
            # jacobian_init=torch.zeros((t_pos_idx.shape[0],9)).to(self.device)
            # jacobian_init[:,0]=1.0
            # jacobian_init[:,4]=1.0
            # jacobian_init[:,8]=1.0
            self.face_attribute = torch.nn.parameter.Parameter(
                jacobian_init, requires_grad=True
            )

            global_dis_init = torch.zeros((3,)).to(self.device)
            self.global_dis = torch.nn.parameter.Parameter(
                global_dis_init, requires_grad=True
            )

            self.register_buffer(
                "v_buffer",
                v_pos,
            )
            self.register_buffer(
                "t_buffer",
                t_pos_idx,
            )

        else:
            raise ValueError(
                f"Unknown shape initialization type: {self.cfg.shape_init}"
            )
        print(self.mesh.v_pos.device)

    def isosurface(self) -> Mesh:
        if hasattr(self, "mesh"):
            return self.mesh
        elif hasattr(self, "v_buffer"):
            self.mesh = Mesh(v_pos=self.v_buffer, t_pos_idx=self.t_buffer)
            return self.mesh
        else:
            raise ValueError(f"custom mesh is not initialized")

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        assert (
            output_normal == False
        ), f"Normal output is not supported for {self.__class__.__name__}"
        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(points, self.bbox)  # points normalized to (0, 1)
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        return {"features": features}

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out

    