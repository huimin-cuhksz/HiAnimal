from dataclasses import dataclass, field

import torch.nn.functional
import trimesh

import threestudio
from threestudio.systems.base import BaseLift3DSystem,BaseSurfaceSystem
from threestudio.utils.typing import *
import pdb
import numpy as np
from pytorch3d.loss import chamfer_distance

def compute_tv_loss(deform_field,faces):
    deform_v0 = deform_field[faces[:, 0]]
    deform_v1 = deform_field[faces[:, 1]]
    deform_v2 = deform_field[faces[:, 2]]

    tv_loss=torch.abs((deform_v0-deform_v1))+torch.abs((deform_v2-deform_v0))+torch.abs((deform_v1-deform_v2))
    tv_loss=torch.mean(torch.sum(torch.sum(tv_loss,dim=2),dim=1))
    return tv_loss

@threestudio.register("deform-fitting-system")
class DeformFittingSystem(BaseSurfaceSystem):
    @dataclass
    class Config(BaseSurfaceSystem.Config):
        refinement: bool = False

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
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
        #map_mask=batch["map_mask"]
        # conf_map=batch['conf_map']
        # conf_mask=(conf_map>0.0).float()
        tgt_fg_mask=batch['tgt_fg_mask']
        tgt_hard_mask=tgt_fg_mask.clone()
        tgt_hard_mask[tgt_hard_mask<1.0]=0.0
        src_hard_mask=batch['src_fg_mask'].clone()
        src_hard_mask[src_hard_mask<1.0]=0.0
        deform_field = out['deform_field']
        gt_deform_field=batch['deform_field']
        deform_mat = out['deform_mat']
        similarity=batch['similarity']
        batch['tgt_normal']=batch['tgt_normal']/2+0.5
        #similarity[similarity<0.1]=0.0
        valid_mask=(similarity>0.1).float()
        #print(deform_field.shape,gt_deform_field.shape)
        deform_loss=torch.nn.functional.l1_loss(deform_field*valid_mask[:,:,None],gt_deform_field*valid_mask[:,:,None])

        #l1_loss=torch.nn.functional.l1_loss(out['comp_rgb']*conf_mask,batch["deform_map"]*conf_mask)
        mask_loss=torch.nn.functional.l1_loss(out["deform_opacity"],batch["tgt_fg_mask"].unsqueeze(3))
        depth_loss=torch.nn.functional.l1_loss(out['deform_depth']*tgt_hard_mask.unsqueeze(3),batch['tgt_depth'].unsqueeze(3)*tgt_hard_mask.unsqueeze(3))
        normal_loss=torch.nn.functional.l1_loss(out['deform_normal']*tgt_hard_mask.unsqueeze(3),batch['tgt_normal']*tgt_hard_mask.unsqueeze(3))
        albedo_loss=torch.nn.functional.l1_loss(out['albedo'],batch['albedo']*src_hard_mask.unsqueeze(3))

        faces=out['faces']
        #print(deform_mat[0,])
        tv_loss=compute_tv_loss(deform_mat,faces)

        deformed_vert=out['verts']+deform_field
        #pdb.set_trace()
        cd_loss_all,cd_loss_normal=chamfer_distance(deformed_vert.unsqueeze(0),batch["tgt_vert"],batch_reduction=None,point_reduction=None)
        src2tgt_dist,tgt2src_dist=cd_loss_all[0],cd_loss_all[1]
        src2tgt_dist[src2tgt_dist>0.1]=0
        tgt2src_dist[tgt2src_dist>0.1]=0
        cd_loss=(torch.mean(src2tgt_dist)+torch.mean(tgt2src_dist))/2

        max_lambda_tv = self.cfg.loss['max_lambda_tv']
        min_lambda_tv = self.cfg.loss['min_lambda_tv']
        max_steps = self.cfg.loss['decay_steps']
        current_steps = self.true_global_step
        lambda_tv = (max_lambda_tv - min_lambda_tv) * max(0, max_steps - current_steps) / max_steps + min_lambda_tv

        lambda_depth=self.cfg.loss['max_lambda_depth'] * max(0, max_steps - current_steps) / max_steps
        lambda_normal = self.cfg.loss['max_lambda_normal'] * max(0, max_steps - current_steps) / max_steps
        lambda_mask = self.cfg.loss['max_lambda_mask'] * max(0, max_steps - current_steps) / max_steps

        loss += deform_loss*self.C(self.cfg.loss["lambda_deform"]) + \
                albedo_loss * self.C(self.cfg.loss['lambda_albedo']) + \
                cd_loss * self.C(self.cfg.loss['lambda_cd']) + \
                tv_loss * self.C(lambda_tv) + \
                mask_loss*self.C(lambda_mask) + \
                depth_loss*self.C(lambda_depth) + \
                normal_loss*self.C(lambda_normal)

        self.log("train/deform_loss", deform_loss)
        # self.log("train/mask_loss", mask_loss)
        # self.log("train/depth_loss", depth_loss)
        # self.log("train/normal_loss", normal_loss)
        self.log("train/cd_loss",cd_loss)
        self.log("train/tv_loss", tv_loss)
        self.log("train/lambda_tv",lambda_tv)

        # for name, value in guidance_out.items():
        #     self.log(f"train/{name}", value)
        #     if name.startswith("loss_"):
        #         loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        #
        # for name, value in self.cfg.loss.items():
        #     self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        batch['tgt_normal'] = batch['tgt_normal'] / 2 + 0.5
        tgt_fg_mask = batch['tgt_fg_mask']
        hard_mask = tgt_fg_mask.clone()
        hard_mask[hard_mask < 1.0] = 0.0
        batch.update({'hard_mask': hard_mask})
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC","data_range":(-1,1)},
                },
            ]
            +
            [
                {
                    "type": "rgb",
                    "img": out["deform_albedo"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            +
            [
                {
                    "type": "grayscale",
                    "img": out["deform_opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None,"data_range": (0, 1)},
                }
            ]
            +
            [
                {
                    "type": "rgb",
                    "img": out["deform_normal"][0, :, :],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                }
            ]
            +
            [
                {
                    "type": "rgb",
                    "img": batch["tgt_normal"][0, :, :],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                }
            ]
            +
            [
                {
                    "type": "grayscale",
                    "img": out["deform_depth"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 5)},
                }
            ]
            +
            [
                {
                    "type": "grayscale",
                    "img": batch["tgt_depth"][0, :, :],
                    "kwargs": {"cmap": None, "data_range": (0, 5)},
                }
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

        deform_field=out['deform_field']
        verts=out['verts']
        faces=out['faces']
        min_max=batch['min_max']
        deform_vert=deform_field+verts
        deform_field=((deform_field-min_max[0,0])/(min_max[0,1]-min_max[0,0])).cpu().numpy()*255.0
        deform_field=deform_field.astype(np.uint8)

        mesh=trimesh.Trimesh(vertices=verts.cpu().numpy(),faces=faces.cpu().numpy().astype(np.int32))
        mesh.visual.vertex_colors=deform_field

        deform_mesh=trimesh.Trimesh(vertices=deform_vert.cpu().numpy(),faces=faces.cpu().numpy().astype(np.int32))

        save_filename="deform_field_mesh.obj"
        mesh_save_path=self.get_save_path(save_filename)
        mesh.export(mesh_save_path)

        deform_save_name="deform_mesh.obj"
        deform_save_path=self.get_save_path(deform_save_name)
        deform_mesh.export(deform_save_path)

        src_points=verts.cpu().numpy()
        tgt_points=deform_vert.cpu().numpy()
        deform_field_save_name="deform_field_3d.npz"
        deform_field_save_path=self.get_save_path(deform_field_save_name)
        np.savez_compressed(deform_field_save_path,src_points=src_points,tgt_points=tgt_points)



    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
