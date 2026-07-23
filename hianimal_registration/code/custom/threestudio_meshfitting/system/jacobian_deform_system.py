from dataclasses import dataclass, field

import torch.nn.functional
import trimesh

import threestudio
from threestudio.systems.base import BaseLift3DSystem, BaseSurfaceSystem
from threestudio.utils.typing import *
import pdb
import numpy as np
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
from pytorch3d.structures import Meshes
import os


def compute_tv_loss(deform_field, faces):
    deform_v0 = deform_field[faces[:, 0]]
    deform_v1 = deform_field[faces[:, 1]]
    deform_v2 = deform_field[faces[:, 2]]

    tv_loss = (
        torch.abs((deform_v0 - deform_v1))
        + torch.abs((deform_v2 - deform_v0))
        + torch.abs((deform_v1 - deform_v2))
    )
    tv_loss = torch.mean(torch.sum(torch.sum(tv_loss, dim=2), dim=1))
    return tv_loss


@threestudio.register("jacobian-deform-system")
class DeformFittingSystem(BaseSurfaceSystem):
    @dataclass
    class Config(BaseSurfaceSystem.Config):
        refinement: bool = False
        debug: bool = True

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.template_mesh = Meshes(
            verts=[self.geometry.mesh.v_pos.to(self.device)],
            faces=[self.geometry.mesh.t_pos_idx.to(self.device)]
        ).to(self.device)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        mesh = self.geometry.isosurface()
        jacobian_pred = self.geometry.face_attribute[:, 0:9].reshape(-1, 3, 3)
        deform_vert = self.geometry.src_jacobian.vertices_from_jacobians(
            jacobian_pred.unsqueeze(0)
        ).squeeze(0)
        deform_vert = deform_vert + self.geometry.global_dis
        return {
            "verts": mesh.v_pos,
            "faces": mesh.t_pos_idx,
            "deform_field": deform_vert - mesh.v_pos,
            "pred_jacobian": jacobian_pred,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        # self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
        #     self.cfg.prompt_processor
        # )
        # self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = 0.0
        if self.cfg.debug and self.true_global_step % 1000 == 0:
            src_vert = out["verts"].detach().cpu().numpy()
            pred_deform_field = out["deform_field"].detach().cpu().numpy()
            target_deform_field = batch["deform_field"].detach().cpu().numpy()
            tgt_vert = batch["tgt_vert"].detach().cpu().numpy()
            deform_field_mask = batch["deform_field_mask"].detach().cpu().numpy()
            debug_filename = f"train-it{self.true_global_step}-debug.npz"
            debug_filepath = self.get_save_path(debug_filename)
            np.savez_compressed(
                debug_filepath,
                src_vert=src_vert,
                pred_deform_field=pred_deform_field,
                target_deform_field=target_deform_field,
                deform_field_mask=deform_field_mask,
                tgt_vert=tgt_vert,
            )
        # map_mask=batch["map_mask"]
        # conf_map=batch['conf_map']
        # conf_mask=(conf_map>0.0).float()

        # tgt_fg_mask = batch["tgt_fg_mask"]
        # tgt_hard_mask = tgt_fg_mask.clone()
        # tgt_hard_mask[tgt_hard_mask < 1.0] = 0.0
        # src_hard_mask = batch["src_fg_mask"].clone()
        # src_hard_mask[src_hard_mask < 1.0] = 0.0

        deform_field = out["deform_field"]
        gt_deform_field = batch["deform_field"]
        # similarity = batch["similarity"]
        deform_field_mask = batch["deform_field_mask"]

        # valid_mask = (similarity > 0.1).float()
        # valid_mask = valid_mask[0]
        # valid_ind = torch.where(valid_mask > 0)[0]

        # valid_ind = torch.where(deform_field_mask)[0]
        # valid_ind = deform_field_mask.long()
        # perm = torch.randperm(valid_ind.shape[0])
        # choose_ind = valid_ind[perm[0:100]]
        perm = torch.randperm(deform_field_mask.shape[0])
        choose_ind = perm[0:100]
        
        deformed_vert = out["verts"][deform_field_mask[:, 1]] + deform_field[deform_field_mask[:, 1]]
        gt_vert = out["verts"][deform_field_mask[:, 1]] + gt_deform_field

        # deformed_vert = deformed_vert-deformed_vert.mean(0,keepdim=True)+gt_vert.mean(0,keepdim=True)

        deform_loss = torch.nn.functional.mse_loss(
            deformed_vert[choose_ind], gt_vert[choose_ind]
        )

        # l1_loss=torch.nn.functional.l1_loss(out['comp_rgb']*conf_mask,batch["deform_map"]*conf_mask)
        mask_loss = torch.zeros((), device=self.device)
        # depth_loss=torch.nn.functional.l1_loss(out['deform_depth']*tgt_hard_mask.unsqueeze(3),batch['tgt_depth'].unsqueeze(3)*tgt_hard_mask.unsqueeze(3))
        # normal_loss=torch.nn.functional.l1_loss(out['deform_normal']*tgt_hard_mask.unsqueeze(3),batch['tgt_normal']*tgt_hard_mask.unsqueeze(3))

        # albedo_loss = torch.nn.functional.l1_loss(
        #     out["albedo"], batch["albedo"] * src_hard_mask.unsqueeze(3)
        # )
        # albedo_loss = torch.nn.functional.l1_loss(
        #     out["albedo"], batch["albedo"]
        # )

        cd_loss_all, cd_loss_normal = chamfer_distance(
            deformed_vert.unsqueeze(0),
            batch["tgt_vert"],
            batch_reduction=None,
            point_reduction=None,
        )
        src2tgt_dist, tgt2src_dist = cd_loss_all[0], cd_loss_all[1]
        # src2tgt_dist[src2tgt_dist>0.1]=0
        # tgt2src_dist[tgt2src_dist>0.1]=0
        cd_loss = (torch.mean(src2tgt_dist) + torch.mean(tgt2src_dist)) / 2

        pred_jacobian = out["pred_jacobian"]
        eye_jacobian = (
            torch.eye(3)
            .unsqueeze(0)
            .expand(pred_jacobian.shape[0], -1, -1)
            .float()
            .to(pred_jacobian.device)
        )
        # print(pred_jacobian.shape,eye_jacobian.shape)
        reg_loss = torch.nn.functional.mse_loss(pred_jacobian, eye_jacobian)

        max_lambda_deform = self.cfg.loss["max_lambda_deform"]
        min_lambda_deform = self.cfg.loss["min_lambda_deform"]
        max_steps = self.cfg.loss["decay_steps"]
        current_steps = self.true_global_step
        lambda_deform = (max_lambda_deform - min_lambda_deform) * max(
            0, max_steps - current_steps
        ) / max_steps + min_lambda_deform

        max_lambda_reg = self.cfg.loss["max_lambda_reg"]
        min_lambda_reg = self.cfg.loss["min_lambda_reg"]
        max_steps = self.cfg.loss["decay_steps"]
        current_steps = self.true_global_step
        lambda_reg = (max_lambda_reg - min_lambda_reg) * max(
            0, max_steps - current_steps
        ) / max_steps + min_lambda_reg

        # 确保变形顶点在正确的设备上，并且形状正确
        deformed_vert = (out["verts"] + deform_field).to(self.device)
        deformed_vert = deformed_vert.unsqueeze(0)  # 添加批次维度
        

        self.template_mesh = self.template_mesh.to(self.device)
        
        deformed_mesh = self.template_mesh.update_padded(deformed_vert)
        laplacian_loss = mesh_laplacian_smoothing(deformed_mesh)

        loss += (
            deform_loss * self.C(lambda_deform)
            # + albedo_loss * self.C(self.cfg.loss["lambda_albedo"])
            + cd_loss * self.C(self.cfg.loss["lambda_cd"])
            + mask_loss * self.C(self.cfg.loss["lambda_mask"])
            + reg_loss * self.C(lambda_reg)
            + laplacian_loss * 0.05  # 拉普拉斯损失权重
        )

        self.log("train/deform_loss", deform_loss)
        self.log("train/cd_loss", cd_loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        return None

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        # self.save_image_grid(
        #     f"it{self.true_global_step}-test/{batch['index'][0]}.png",
        #     # [
        #     #     {
        #     #         "type": "rgb",
        #     #         "img": out["comp_rgb"][0],
        #     #         "kwargs": {"data_format": "HWC"},
        #     #     },
        #     # ]
        #     (
        #         [
        #             {
        #                 "type": "rgb",
        #                 "img": out["comp_normal"][0],
        #                 "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
        #             }
        #         ]
        #         if "comp_normal" in out
        #         else []
        #     )
        #     + [
        #         {
        #             "type": "grayscale",
        #             "img": out["opacity"][0, :, :, 0],
        #             "kwargs": {"cmap": None, "data_range": (0, 1)},
        #         },
        #     ],
        #     name="test_step",
        #     step=self.true_global_step,
        # )

        deform_field = out["deform_field"]
        deform_field_mask = batch["deform_field_mask"]
        verts = out["verts"]
        faces = out["faces"]
        # min_max = batch["min_max"]
        min_, max_ = deform_field.min(), deform_field.max()
        # min_, max_ = deform_field[deform_field_mask].min(), deform_field[deform_field_mask].max()
        deform_vert = deform_field + verts
        deform_field = (
            (deform_field - min_) / (max_ - min_)
        ).cpu().numpy() * 255.0
        # deform_field = (
        #     (deform_field - min_max[0, 0]) / (min_max[0, 1] - min_max[0, 0])
        # ).cpu().numpy() * 255.0
        deform_field = deform_field.astype(np.uint8)

        mesh = trimesh.Trimesh(
            vertices=verts.cpu().numpy(), faces=faces.cpu().numpy().astype(np.int32)
        )
        mesh.visual.vertex_colors = deform_field

        deform_mesh = trimesh.Trimesh(
            vertices=deform_vert.cpu().numpy(),
            faces=faces.cpu().numpy().astype(np.int32),
        )

        save_filename = "deform_field_mesh.obj"
        mesh_save_path = self.get_save_path(save_filename)
        mesh.export(mesh_save_path)

        deform_save_name = "deform_mesh.obj"
        deform_save_path = self.get_save_path(deform_save_name)
        deform_mesh.export(deform_save_path)

        src_points = verts.cpu().numpy()
        tgt_points = deform_vert.cpu().numpy()
        deform_field_save_name = "deform_field_3d.npz"
        deform_field_save_path = self.get_save_path(deform_field_save_name)
        np.savez_compressed(
            deform_field_save_path, src_points=src_points, tgt_points=tgt_points
        )

        print(f"Saved deform_field_mesh.obj to: {mesh_save_path}")
        print(f"Saved deform_mesh.obj to: {deform_save_path}")

    def on_test_epoch_end(self):
        pass
