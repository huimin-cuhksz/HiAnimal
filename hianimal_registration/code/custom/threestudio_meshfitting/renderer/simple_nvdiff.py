from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F
# from kornia.geometry import depth

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *
from threestudio.utils.ops import dot
import pdb

def compute_vertex_normal(verts,faces):
    i0 = faces[:, 0]
    i1 = faces[:, 1]
    i2 = faces[:, 2]

    v0 = verts[i0, :]
    v1 = verts[i1, :]
    v2 = verts[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(verts)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(
        dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
    )
    v_nrm = F.normalize(v_nrm, dim=1)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))

    return v_nrm


@threestudio.register("simple-nvdiff-rasterizer")
class NVDiffRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    def forward(
        self,
        src_mtx: Float[Tensor, "B 4 4"],
        tgt_mtx: Float[Tensor, "B 4 4"],
        src_camera_positions: Float[Tensor, "B 3"],
        tgt_camera_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        render_rgb: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = src_mtx.shape[0]
        mesh = self.geometry.isosurface()

        #min_max=kwargs["min_max"]
        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, src_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        '''render deform mask, depth and normal'''
        #deform_out=self.geometry(mesh.v_pos,output_normal=False)
        deform_mat=self.geometry.vertex_attribute[:,0:12]
        deform_mat=deform_mat.reshape(-1,4,3)
        #pdb.set_trace()
        org_vert=mesh.v_pos
        homo_vert=torch.cat([mesh.v_pos,torch.ones((mesh.v_pos.shape[0],1)).to(mesh.v_pos.device)],dim=1)
        deform_vert=torch.einsum("nk,nkj->nj",homo_vert,deform_mat)
        deform_field=deform_vert-org_vert

        # deform_field = self.material(
        #     **deform_out
        # )
        #print(torch.amin(deform_field),torch.amax(deform_field))
        #deform_field_w=deform_field*(min_max[0,1]-min_max[0,0])+min_max[0,0] #convert it to the original scale
        #print(torch.min(deform_field_w),torch.max(deform_field_w),min_max[0,0],min_max[0,1])

        out = { #use to compute tv regularization
            "verts":mesh.v_pos,
            "faces": mesh.t_pos_idx,
            "deform_field":deform_field,
            "deform_mat":deform_mat
        }
        deform_vpos_clip=self.ctx.vertex_transform(
            deform_vert, tgt_mtx
        )

        deform_rast, _ = self.ctx.rasterize(deform_vpos_clip, mesh.t_pos_idx, (height, width))
        deform_mask = deform_rast[...,3:]>0
        deform_mask_aa = self.ctx.antialias(deform_mask.float(), deform_rast,
                                            deform_vpos_clip, mesh.t_pos_idx)

        tgt_w2c=kwargs['tgt_w2c']
        deform_v_incam=self.ctx.vertex_transform(
            deform_vert, tgt_w2c
        )
        deform_vz=-deform_v_incam[0,:,2:3]
        #pdb.set_trace()
        deform_depth,_=self.ctx.interpolate_one(deform_vz, deform_rast, mesh.t_pos_idx)
        deform_depth_aa=self.ctx.antialias(deform_depth,deform_rast,deform_vpos_clip,mesh.t_pos_idx)

        deform_v_nrm=compute_vertex_normal(deform_vert,mesh.t_pos_idx)
        #deform_v_nrm=compute_vertex_normal(mesh.v_pos,mesh.t_pos_idx)
        #pdb.set_trace()
        deform_normal,_=self.ctx.interpolate_one(deform_v_nrm,deform_rast, mesh.t_pos_idx)
        gb_normal = F.normalize(deform_normal, dim=-1)
        gb_normal_aa = torch.lerp(
            torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, deform_mask.float()
        )
        gb_normal_aa = self.ctx.antialias(
            gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
        )
        out.update({"opacity": mask_aa, "deform_opacity": deform_mask_aa, "mesh": mesh,
               "deform_depth":deform_depth_aa,"deform_normal":gb_normal_aa})


        '''render albedo map'''
        selector = mask[..., 0]
        gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
        positions = gb_pos[selector]
        geo_out = self.geometry(positions, output_normal=False)
        geo_out["features"]=geo_out["features"][:,0:3]
        albedo = self.material(**geo_out)
        albedo_fg = torch.zeros(batch_size, height, width, 3).to(albedo)
        albedo_fg[selector] = albedo

        '''render deform albedo map'''
        deform_selector = deform_mask[...,0]
        deform_pos, _ =self.ctx.interpolate_one(mesh.v_pos, deform_rast, mesh.t_pos_idx)
        deform_positions = deform_pos[deform_selector]
        deform_color_out=self.geometry(deform_positions,output_normal=False)
        deform_color_out["features"]=deform_color_out["features"][:,0:3]
        deform_albedo=self.material(**deform_color_out)
        deform_albedo_fg=torch.zeros(batch_size,height,width,3).to(deform_albedo)
        deform_albedo_fg[deform_selector]=deform_albedo

        '''render_deformation map'''
        gb_viewdirs = F.normalize(
            gb_pos - src_camera_positions[:, None, None, :], dim=-1
        )
        gb_rgb_fg,_ = self.ctx.interpolate_one(deform_field.contiguous(), rast, mesh.t_pos_idx)

        gb_rgb_bg = self.background(dirs=gb_viewdirs)
        gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())

        albedo_wbg = torch.lerp(gb_rgb_bg,albedo_fg, mask.float())
        albedo_aa = self.ctx.antialias(albedo_wbg, rast, v_pos_clip, mesh.t_pos_idx)
        gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

        out.update({"comp_rgb": gb_rgb,'albedo':albedo_aa, "comp_rgb_bg": gb_rgb_bg,"deform_albedo":deform_albedo_fg})

        return out
