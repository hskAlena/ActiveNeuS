# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of vanilla nerf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type, Literal, Optional

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch.nn.functional as F
import nerfacc

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding, lossfun_occ_reg
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import (
    MSELoss, L1Loss, VarianceLoss
)
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler, NeRFAccSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    UncertaintyRenderer
)
from nerfstudio.model_components.scene_colliders import (
    AABBBoxCollider,
    NearFarCollider,
    SphereCollider,
)
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    RGBFieldHead,
    UncertaintyFieldHead
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.vanilla_nerf import VanillaModelConfig, NeRFModel
from nerfstudio.utils import colormaps, colors, misc
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.utils.colors import get_color
from nerfstudio.fields.base_field import Field, FieldConfig
from nerfstudio.utils.ause import *

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min() + 1e-10)
    depth = torch.clip(depth, 0, 1)
    return depth

@dataclass
class NeRFAccModelConfig(VanillaModelConfig):
    """Vanilla Model Config"""

    _target: Type = field(default_factory=lambda: NeRFAccModel)    
    steps_warmup: int = 256
    steps_per_grid_update: int = 16
    grid_resolution: int = 128

    acquisition: Literal["random", "fvs","entropy", "active", "frontier", 'freenerf'] = "random"
    choose_multi_cam: Literal["dist", "topk"] = "topk"
    entropy_type: Literal[ "ent", "no_surface"] = "no_surface"
    grid_sampling: bool =True
    var_eps=1e-6
    uncertainty_net: bool = False
    beta_min=0.001
    uncert_rgb_loss_mult: float = 0.001
    """sparse point sdf loss multiplier"""
    uncert_beta_loss_mult: float = 0.01  #0.01~0.25
    uncert_sigma_loss_mult: float = 0.0
    fg_mask_loss_mult: float = 0.01
    """Foreground mask loss multiplier."""
    #########################################
    grid_levels: int = 4
    """Levels of the grid used for the field."""
    dist_threshold: float = 1.732
    # nerf_field: NeRFField = NeRFFieldConfig()
    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation"""
    # num_importance_samples: int = 128
    # """Number of samples in fine field evaluation"""
    enable_collider: bool = True
    """Whether to create a scene collider to filter rays."""
    # collider_params: Optional[Dict[str, float]] = None
    collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 0.05, "far_plane": 1e3})
    """Instant NGP doesn't use a collider."""
    disable_scene_contraction: bool = False
    # """Whether to disable scene contraction or not."""
    ######################################
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = None
    """Minimum step size for rendering."""
    near_plane: float = 0.05
    """How far along ray to start sampling."""
    far_plane: float = 1e3
    """How far along ray to stop sampling."""
    background_color: Literal["random", "black", "white"] = "random"
    """The color that is given to untrained areas."""
    alpha_sample_thre: float = 1e-2
    """alpha thres for visibility pruning in nerfacc, should set to 0 for the nerf-synthetic dataset"""
    no_acc: bool =False
    acq_vis_type: Literal["sum", "alpha-composition"] = "sum"


class NeRFAccModel(NeRFModel):
    """Vanilla NeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """
    config: NeRFAccModelConfig

    def __init__(
        self,
        config: NeRFAccModelConfig,
        **kwargs,
    ) -> None:
        self.field = None
        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))
        self.config.collider_params['near_plane'] = self.config.near_plane
        self.config.collider_params['far_plane'] = self.config.far_plane
        if self.config.enable_collider:
            self.collider = NearFarCollider(
                near_plane=self.config.collider_params["near_plane"], far_plane=self.config.collider_params["far_plane"]
            )
        # fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.field = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            use_uncertainty_net=self.config.uncertainty_net,
            spatial_distortion=scene_contraction,
            field_heads = (RGBFieldHead(),DensityFieldHead(), UncertaintyFieldHead()),
            frequency_regularizer= self.config.frequency_regularizer,
            posenc_len=self.config.posenc_len,
            direnc_len=self.config.direnc_len,
            freq_reg_end=self.config.freq_reg_end,
            freq_reg_start=self.config.freq_reg_start
        )
        

        # renderers
        self.background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )
        self.renderer_rgb = RGBRenderer(background_color=self.background_color)
        # renderers
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_uncertainty = UncertaintyRenderer()
        self.renderer_depth = DepthRenderer(method='expected')

        # losses
        self.rgb_loss = MSELoss()
        self.rgb_mask_loss = L1Loss(reduction='sum')
        self.uncert_loss = VarianceLoss(self.config.uncert_rgb_loss_mult,
                                        self.config.uncert_beta_loss_mult,
                                        self.config.uncert_sigma_loss_mult)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity() #normalize=True)

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        # NeRFstudio !!!!!!!!!!!
        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.config.render_step_size = \
                ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / 256
        # Occupancy Grid.

        # Occupancy Grid.
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_coarse_samples)
        # Sampler
        
        self.assist_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )
        density_fn = lambda x, steps: self.field.get_density_pts(x,steps=steps)
        self.sampler = NeRFAccSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=density_fn, # self.density_fn
            uniform_sampler = self.sampler_uniform,
            assist_grid=self.assist_grid,
            scene_aabb=self.scene_aabb,
            steps_warmup=self.config.steps_warmup,
            steps_per_grid_update=self.config.steps_per_grid_update,
            topk_iter=self.topk_iter,
            render_step_size=self.config.render_step_size
        )

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)
        # TODO find out why config.render_

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.sampler.update_step,
            )
        )

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.sampler.update_binary_grid,
                # kwargs={"density_fn": lambda x: self.field.density_fn(x)}, # * self.config.render_step_size,},
            )
        )

        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")

        param_groups["fields"] = list(self.field.parameters())
        # if self.temporal_distortion is not None:
        #     param_groups["temporal_distortion"] = list(self.temporal_distortion.parameters())
        return param_groups

    
    def get_outputs(self, ray_bundle: RayBundle, steps=None):
        if self.config.no_acc or (self.sampler._update_counter.item() <= 0):
            return super().get_outputs(ray_bundle, steps=steps)
        assert self.field is not None
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                alpha_sample_thre=self.config.alpha_sample_thre,
                cone_angle=self.config.cone_angle,
            )


        field_outputs = self.field(ray_samples, steps=steps)
        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)

        # Maybe because of uniform sampler? the dimension is not continuous?

        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )
        # weights, transmittance, alphas
        weights = weights[0]
        # weights = weights[..., None]

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            'sigma': field_outputs[FieldHeadNames.DENSITY],
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
        }
        sigma_dup = torch.repeat_interleave(packed_info[:, -1], packed_info[:, -1], 0)
        sigma_div_up = field_outputs[FieldHeadNames.DENSITY][..., 0]/sigma_dup
        outputs['mean_sigma'] = torch.unsqueeze(torch.sum(sigma_div_up)/num_rays, dim=0)
        if self.training:
            outputs.update({
                "ray_samples": ray_samples,
                'ray_indices': ray_indices,
                'n_rays': num_rays,
                "weights": weights,
            })

        # if steps %self.topk_iter ==0:
        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)
        def make_entropy(val):
            # you can delete 1e-6 when the self.alpha_sample_thre is low enough.
            val = val.clip(1e-6, 1.0- (1e-6))
            return -val*torch.log(val) -(1-val)*torch.log(1-val)

        occupancy_points, occup_mask = nerfacc.grid._query(inputs, 
                                                            self.sampler._alpha,
                                                            self.scene_aabb)
        grid_ent, _ = self.sampler.cal_entropy(inputs, ray_indices, occupancy_points,
                                    make_entropy, 'ent', None, self.assist_grid, self.scene_aabb, mean=False )
        grid_ent = torch.unsqueeze(grid_ent, dim=-1)
        if self.config.acq_vis_type == "alpha-composition":
            pseudo_weight = weights
        else:
            pseudo_weight = torch.ones_like(weights) #/sigma_dup
        
        # Entropys are not normal, but just used normal renderer for visualization
        accum_ent = self.renderer_uncertainty(weights=pseudo_weight, 
                                                    ray_indices=ray_indices,
                                                    num_rays=num_rays,
                                                    betas=grid_ent)
        accum_ent =  torch.where(accum_ent!=0, accum_ent, 0.0)
        outputs.update({
            "accum_entropy": accum_ent,
        })
        #################################################################################
        if self.config.uncertainty_net:
            # uniform sampling
            ray_samples_uniform = self.sampler_uniform(ray_bundle)
            # coarse field:
            density, betas, _ = self.field.get_density(ray_samples_uniform, compute_uncertainty=True, steps=steps)
            weights_coarse = ray_samples_uniform.get_weights(
                density # [1024, 128, 1]
            )
            # pdf sampling
            ray_samples_active = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
            # fine field:
            density, betas, _ = self.field.get_density(ray_samples_active, compute_uncertainty=True, steps=steps)
            weights_active = ray_samples_active.get_weights(
                density # [1024, 128, 1]
            )

            ########################################################
            # betas.shape [1024*128, 1]
            beta_shape = betas.reshape(ray_samples_active.shape[0], -1, 1)
            

            uncertainty_active = self.renderer_uncertainty(weights=weights_active**2, 
                                                        betas=beta_shape)
            var_pri = beta_shape+ self.config.var_eps
            var_samples = torch.unsqueeze(uncertainty_active, dim=1).repeat(1, ray_samples_active.shape[1], 1)
            var_post = 1 / (1 / var_pri + weights_active**2 / (var_samples + self.config.var_eps))
            acq = var_pri - var_post

            if self.config.acq_vis_type == "alpha-composition":
                pseudo_weight = weights_active
                uncertainty_active = self.renderer_uncertainty(weights=pseudo_weight, 
                                                            betas=acq)
            else:
                uncertainty_active = torch.sum(acq, dim=1)
            outputs.update({
                "uncertainty_active": uncertainty_active,
            })

        if self.config.uncertainty_net:
            uncertainty = self.renderer_uncertainty(weights=weights**2, 
                                                ray_indices=ray_indices,
                                                num_rays=num_rays,
                                                betas=field_outputs[FieldHeadNames.UNCERTAINTY])
            # if uncertainty.min()>0:
            outputs.update({
                "uncertainty": uncertainty,
            })
        return outputs
    
    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        metrics_dict = {}
        if 'rgb' in outputs:
            rgb = outputs["rgb"]
        else:
            rgb = outputs["rgb_fine"]
        if "fg_mask" in batch and self.config.fg_mask_loss_mult > 0.0:
            fg_label = batch["fg_mask"].float().to(rgb.device)
            image = fg_label*image
            rgb = fg_label*rgb
        metrics_dict["psnr"] = self.psnr(rgb, image)
        if 'num_samples_per_ray' in outputs:
            metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        if self.sampler._update_counter.item() <= 0:
            return super().get_loss_dict( outputs, batch, metrics_dict=metrics_dict)

        image = batch["image"][..., :3].to(self.device)
        loss_dict = {}
        if 'rgb' in outputs:
            pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
                pred_image=outputs["rgb"],
                pred_accumulation=outputs["accumulation"],
                gt_image=image,
            )
        else:
            pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
                pred_image=outputs["rgb_fine"],
                pred_accumulation=outputs["accumulation_fine"],
                gt_image=image,
            )

        if "fg_mask" in batch and self.config.fg_mask_loss_mult > 0.0:
            fg_label = batch["fg_mask"].float().to(self.device)
            color_error = (pred_rgb-image)*fg_label
            mask_sum = fg_label.sum() + 1e-5
            loss_dict['rgb_loss'] = self.rgb_mask_loss(color_error, 
                                                       torch.zeros_like(color_error)) /mask_sum
        else:
            loss_dict["rgb_loss"] = self.rgb_loss(image, pred_rgb)

        # if self.config.occ_reg_loss_mult > 0.0:
        #     rgb = outputs['rgb']
        #     density = outputs['sigma']
        #     occ_reg_loss = torch.mean(lossfun_occ_reg(
        #         rgb, density, reg_range=self.config.occ_reg_range,
        #         # wb means white&black prior in DTU
        #         wb_prior=self.config.occ_wb_prior, wb_range=self.config.occ_wb_range)) 
        #     occ_reg_loss = self.config.occ_reg_loss_mult * occ_reg_loss
        #     loss_dict['occ_reg_loss']= occ_reg_loss

        if "uncertainty" in outputs and (
            self.config.uncert_rgb_loss_mult > 0.0
            or self.config.uncert_beta_loss_mult > 0.0
            or self.config.uncert_sigma_loss_mult > 0.0
        ):
            if "fg_mask" in batch and self.config.fg_mask_loss_mult > 0.0:
                fg_mask_flag = True
            else:
                fg_mask_flag = False
                fg_label = None
            uncert_dict = self.uncert_loss(image, outputs, fg_flag=fg_mask_flag, fg_mask=fg_label)
            loss_dict['uncertainty_rgb'] = uncert_dict['rgb']
            loss_dict['uncertainty_loss'] = uncert_dict['uncertainty']
            if 'density' in uncert_dict.keys():
                loss_dict['uncertainty_density'] = uncert_dict['density']
            # loss_dict["uncertainty_loss"] = (
            #     uncert_dict['rgb'] + uncert_dict['uncertainty'] #uncert_dict['sigma']
            # )
        

        # foreground mask loss
        if "fg_mask" in batch and self.config.fg_mask_loss_mult > 0.0:
            ## TODO may need to mask rgb_loss as well
            if 'weights' not in outputs or fg_label.shape[0] != outputs['weights'].shape[0]:
                if 'accumulation' in outputs:
                    accum = outputs['accumulation'].clip(1e-3, 1.0 - 1e-3)
                    loss_dict["fg_mask_loss"] = (
                        F.binary_cross_entropy(accum, fg_label) * self.config.fg_mask_loss_mult
                    )
                else:
                    accum = outputs['accumulation_fine'].clip(1e-3, 1.0 - 1e-3)
                    loss_dict["fg_mask_loss"] = (
                        F.binary_cross_entropy(accum, fg_label) * self.config.fg_mask_loss_mult
                    )
            else:
                weights_sum = outputs["weights"].sum(dim=1).clip(1e-3, 1.0 - 1e-3)
                loss_dict["fg_mask_loss"] = (
                    F.binary_cross_entropy(weights_sum, fg_label) * self.config.fg_mask_loss_mult
                )
        return loss_dict

    
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                                            number=0, datapath='./data/', cameras=None
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        if 'rgb' in outputs:
            rgb = outputs["rgb"]
            acc = colormaps.apply_colormap(outputs["accumulation"])
            depth = colormaps.apply_depth_colormap(
                outputs["depth"],
                accumulation=outputs["accumulation"],
            )
        else:
            rgb = outputs["rgb_fine"]
            acc = colormaps.apply_colormap(outputs["accumulation_fine"])
            depth = colormaps.apply_depth_colormap(
                outputs["depth_fine"],
                accumulation=outputs["accumulation_fine"],
            )
        # import pdb
        # pdb.set_trace()
        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        if "fg_mask" in batch and self.config.fg_mask_loss_mult > 0.0:
            fg_label = batch["fg_mask"].float().to(rgb.device)
            image = fg_label*image + (1-fg_label)
            rgb = fg_label*rgb + (1-fg_label)
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips((image*2-1), (rgb*2-1))

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        if 'uncertainty' in outputs:
            uncertainty = depth2img(outputs['uncertainty'])
            uncertainty = colormaps.apply_colormap(uncertainty)
            images_dict['uncertainty'] = uncertainty
            if 'dtu' in datapath:
                unc_metric_dict, unc_images_dict = get_image_metrics_unc(self, number, outputs, batch, datapath, cameras)
                images_dict.update(unc_images_dict)
                metrics_dict.update(unc_metric_dict)
        if 'accum_entropy' in outputs:
            accum_entropy = visualize_rank_unc(outputs['accum_entropy'].squeeze(-1).cpu().numpy())
            images_dict['accum_entropy'] = torch.from_numpy(accum_entropy[...,:-1]) 
        if 'uncertainty_active' in outputs:
            uncertainty_active = visualize_rank_unc(outputs['uncertainty_active'].squeeze(-1).cpu().numpy())
            images_dict['uncertainty_active'] =  torch.from_numpy(uncertainty_active[...,:-1]) 
            # uncertainty_active = visualize_rank_unc(outputs['uncertainty_active_alpha'].squeeze(-1).cpu().numpy())
            # images_dict['uncertainty_active_alpha'] =  torch.from_numpy(uncertainty_active[...,:-1])

        return metrics_dict, images_dict
    
    def initialize_grid(self, camera_info, i_train, transform=True):
        camera_to_worlds = camera_info.camera_to_worlds[i_train].to(self.device)
        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        # only for dtu case, not blender
        if transform:
            camera_to_worlds[:, 0:3, 1:3] *= -1
        fx = torch.mean(camera_info.fx[i_train])
        fy = torch.mean(camera_info.fy[i_train])
        cx = torch.mean(camera_info.cx[i_train])
        cy = torch.mean(camera_info.cy[i_train])
        width = camera_info.width[i_train][0].to(self.device)
        height = camera_info.height[i_train][0].to(self.device)
        K = torch.Tensor([[fx, 0, width/2],
                            [0, fy, height/2],
                            [0,  0,   1]])
        K = torch.unsqueeze(K, dim=0).to(self.device)
        filtered_info = {
            'K': K,
            'c2w': camera_to_worlds,
            'width': width,
            'height':  height,
        }

        self.sampler.initialize_grid(filtered_info, chunk=32**3, near_plane=self.config.near_plane)
    
    def eval_k_views(self, ray_bundle):
        # origins, directions, pixel_area, directions_norm, camera_indices,
        # nears, fars, metadata, times
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        topk, mean_uncertainty = self.sampler.eval_k_views(ray_bundle,
                                        near_plane=self.config.near_plane,
                                        far_plane=self.config.far_plane,
                                        render_step_size=self.config.render_step_size,
                                        alpha_sample_thre=self.config.alpha_sample_thre,
                                        cone_angle=self.config.cone_angle,
                                        type=self.config.entropy_type,
                                        grid_sampling=self.config.grid_sampling)

            

        return topk, mean_uncertainty
    
    def choose_k_views(self, train_set, candidates, topk_list, candidate_cam_info, k=4):
        if len(candidates) <= k:
            maxidx_pose = candidates
        else:
            topk_list = torch.tensor(topk_list)
            maxidx_pose = self.sampler.cal_multiCam(train_set, candidates, topk_list, candidate_cam_info,
                                                  dist_threshold=self.config.dist_threshold, 
                                                  method=self.config.choose_multi_cam, cand_topk=k)
            # maxidx_pose = torch.topk(self.model.acquisition_grid, self.hparams.topk, dim=0)[1].cpu().numpy()

        return maxidx_pose
    
    @torch.no_grad()
    def active_eval(self, ray_bundle, steps=None):
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)
        # # TODO sampler_uniform should be replaced by sampler_nerf 
        # ray_samples = self.sampler_uniform(ray_bundle) # 1024, 128

        # density, betas, _ = self.field.get_density(ray_samples, compute_uncertainty=True, steps=steps)
        # weights = ray_samples.get_weights(
        #     density # [1024, 128, 1]
        # )

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        # coarse field:
        density, betas, _ = self.field.get_density(ray_samples_uniform, compute_uncertainty=True, steps=steps)
        weights_coarse = ray_samples_uniform.get_weights(
            density # [1024, 128, 1]
        )
        # pdf sampling
        ray_samples = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        # fine field:
        density, betas, _ = self.field.get_density(ray_samples, compute_uncertainty=True, steps=steps)
        weights = ray_samples.get_weights(
            density # [1024, 128, 1]
        )

        # betas.shape [1024*128, 1]
        beta_shape = betas.reshape(ray_samples.shape[0], -1, 1)
        

        uncertainty = self.renderer_uncertainty(weights=weights**2, 
                                                    betas=beta_shape)
        var_pri = beta_shape+ self.config.var_eps
        var_samples = torch.unsqueeze(uncertainty, dim=1).repeat(1, ray_samples.shape[1], 1)
        var_post = 1 / (1 / var_pri + weights**2 / (var_samples + self.config.var_eps))
        acq = var_pri - var_post
        acq = torch.unsqueeze(torch.sum(acq), dim=-1).detach().cpu()

        return acq

    
    def active_iter(self):
        return self.config.acquisition
