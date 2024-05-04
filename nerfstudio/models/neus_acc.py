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
Implementation of VolSDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type, Literal, Optional

import nerfacc
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import NeuSAccSampler, NeuSSampler, UniformSampler
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from torch.nn import Parameter
from nerfstudio.utils.colors import get_color

@dataclass
class NeuSAccModelConfig(NeuSModelConfig):
    """UniSurf Model Config"""

    _target: Type = field(default_factory=lambda: NeuSAccModel)
    sky_loss_mult: float = 0.01
    """Sky segmentation normal consistency loss multiplier."""
    grid_resolution: int = 128
    grid_bg_resolution: int=64
    steps_warmup: int = 256
    steps_per_grid_update: int = 16
    acquisition: Literal["random", "fvs","entropy", 'freenerf'] = "random"
    entropy_type: Literal[ "ent", "no_surface"] = "no_surface"
    choose_multi_cam: Literal["dist", "topk"] = "topk"
    grid_sampling: bool =True

    grid_levels: int = 1
    """Levels of the grid used for the field."""
    dist_threshold: float = 1.732
    num_coarse_samples: int = 64
    overwrite_near_far_plane: bool = True
    """whether to use near and far collider from command line"""
    near_plane: float = 0.05
    """How far along ray to start sampling."""
    far_plane: float = 1e3
    """How far along ray to stop sampling."""
    scene_contraction_norm: Literal["inf", "l2"] = "inf"
    """Which norm to use for the scene contraction."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = None
    """Minimum step size for rendering."""
    alpha_sample_thre: float = 0.01
    """alpha thres for visibility pruning in nerfacc, should set to 0 for the nerf-synthetic dataset"""
    sphere_masking:bool=True
    maintain_aabb: bool=True
    acq_vis_type: Literal["sum", "alpha-composition"] = "sum"

class NeuSAccModel(NeuSModel):
    """VolSDF model

    Args:
        config: MonoSDF configuration to instantiate model
    """

    config: NeuSAccModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)
        self.lpips = LearnedPerceptualImagePatchSimilarity() #normalize=True)
        # NeRFstudio !!!!!!!!!!!
        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.bg_render_step_size = None
        else:
            self.bg_render_step_size = self.config.render_step_size
            # self.config.render_step_size = \
            #     ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / 256
            # self.bg_render_step_size = \
            #     ((self.bg_aabb[3:] - self.bg_aabb[:3]) ** 2).sum().sqrt().item() / 256
        # Occupancy Grid.
        self.grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=1,
        )
        self.assist_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=1,
        )
        if self.config.background_model !='none':
            if not self.config.maintain_aabb:
                bg_aabb =nerfacc.grid._enlarge_aabb(self.scene_box.aabb.flatten(), 4)
                self.bg_aabb = Parameter(bg_aabb, requires_grad=False)
            else:
                self.bg_aabb = self.scene_aabb
            # self.bg_grid = nerfacc.OccupancyGrid(
            self.bg_grid = nerfacc.OccGridEstimator(
                roi_aabb=self.bg_aabb,
                resolution=self.config.grid_bg_resolution,
                levels=self.config.grid_levels,
            )
            self.bg_assist_grid = nerfacc.OccGridEstimator(
                roi_aabb=self.bg_aabb,
                resolution=self.config.grid_bg_resolution,
                levels=self.config.grid_levels,
            )
        else:
            self.bg_aabb = self.scene_aabb
            self.bg_grid = None
            self.bg_assist_grid = None

        self.background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )
        self.neus_sampler = self.sampler
        # voxel surface bybrid sampler from NeuralReconW
        if self.config.background_model in ['grid', 'mlp']:
            background_model = lambda x, steps: self.field_background.get_density_pts(x,steps=steps) #* self.config.render_step_size
        else:
            background_model = None

        self.sampler = NeuSAccSampler(
            aabb=self.scene_box.aabb, 
            grid = self.grid,
            bg_grid= self.bg_grid,
            assist_grid=self.assist_grid,
            bg_assist_grid=self.bg_assist_grid,
            scene_aabb = self.scene_aabb,
            bg_aabb =self.bg_aabb,

            neus_sampler=self.neus_sampler,
            resolution=self.config.grid_resolution,
            bg_resolution=self.config.grid_bg_resolution,

            coarse_sample= self.config.num_coarse_samples,
            topk_iter = self.topk_iter,
            steps_warmup=self.config.steps_warmup,
            steps_per_grid_update=self.config.steps_per_grid_update,

            density_fn=background_model,
            sdf_fn=lambda x, steps: self.field.forward_geonetwork(x,steps=steps)[:, 0].contiguous(),
            inv_s_fn=self.field.deviation_network.get_variance
            # density_fn=self.field.get_density_pts, # self.density_fn
            )


    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        # add sampler call backs
        sdf_fn = lambda x, steps: self.field.forward_geonetwork(x, steps=steps)[:, 0].contiguous()
        inv_s = self.field.deviation_network.get_variance

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.sampler.update_binary_grid,
                # kwargs={"sdf_fn": sdf_fn, "inv_s_fn": inv_s, "bg_model":background_model},
            )
        )

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.sampler.update_step_size,
                kwargs={"inv_s": inv_s},
            )
        )

        return callbacks

    def get_outputs(self, ray_bundle: RayBundle, steps=None):
        # bootstrap with original Neus
        if self.sampler._update_counter.item() <= 0:
            return super().get_outputs(ray_bundle, steps=steps)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                                            ray_bundle=ray_bundle,
                                            near_plane=self.config.near_plane,
                                            far_plane=self.config.far_plane,
                                            render_step_size=self.config.render_step_size,
                                            alpha_sample_thre=0.0,
                                            cone_angle=0.0,
                                        )
            if self.config.background_model in ['grid', 'mlp']:
                bg_ray_samples, bg_ray_indices = self.sampler(
                                        ray_bundle=ray_bundle,
                                        near_plane=self.config.near_plane,
                                        far_plane=self.config.far_plane,
                                        render_step_size=self.bg_render_step_size,
                                        alpha_sample_thre=self.config.alpha_sample_thre,
                                        cone_angle=self.config.cone_angle,
                                        bg_model = True
                                    )
        
        n_rays = ray_bundle.shape[0]
        field_outputs = self.field(ray_samples, return_alphas=True, 
                                   return_occupancy=True, steps=steps)
        packed_info = nerfacc.pack_info(ray_indices, n_rays)
        if self.config.background_model != "none" and self.config.sphere_masking:
            field_outputs, bg_field_outputs, sphere_mask, bg_sphere_mask = self.forward_background_field_and_merge_extend(
                                                        bg_ray_samples=bg_ray_samples, 
                                                           ray_samples=ray_samples,
                                                           field_outputs=field_outputs, 
                                                           compute_uncertainty=False, steps=steps,
                                                           return_sphere_mask=True,)
                                                        #    sphere_aabb=1.74*abs(self.scene_box.aabb[0,0]).to(ray_indices.device))

        weights, transmittance = nerfacc.render_weight_from_alpha(
            field_outputs[FieldHeadNames.ALPHA][...,0],
            packed_info=packed_info,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        # Shotscuts
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], 
                                ray_indices=ray_indices,
                                num_rays=n_rays,
                                weights=weights)
        depth = self.renderer_depth(weights=weights, 
                                    ray_indices=ray_indices,
                                    num_rays=n_rays,
                                    ray_samples=ray_samples)
        depth = depth / ray_bundle.directions_norm
        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], 
                                    ray_indices=ray_indices,
                                num_rays=n_rays,
                                    weights=weights)
        accumulation = self.renderer_accumulation(weights=weights,
                                                    ray_indices=ray_indices,
                                                    num_rays=n_rays)
        ray_points = self.scene_contraction(ray_samples.frustums.get_start_positions())

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
        if self.config.acq_vis_type == "alpha-composition":
            pseudo_weight = weights
        else:
            pseudo_weight = torch.ones_like(weights) #/sigma_dup
        grid_ent = torch.unsqueeze(grid_ent, dim=-1)
        # Entropys are not normal, but just used normal renderer for visualization
        accum_ent = self.renderer_normal(weights=pseudo_weight, 
                                                    ray_indices=ray_indices,
                                                    num_rays=n_rays,
                                                    semantics=grid_ent)
        accum_ent =  torch.where(accum_ent!=0, accum_ent, 0.0)
        
        outputs={
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "num_samples_per_ray": packed_info[:, 1],
            "ray_points": ray_points,  # used for creating visiblity mask
            "directions_norm": ray_bundle.directions_norm,  # used to scale z_vals for free space and sdf loss
        }
        # if steps %self.topk_iter ==0:
        outputs.update({
            "accum_entropy": accum_ent,
        })
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            points_norm = field_outputs["points_norm"]
            outputs.update({"eik_grad": grad_points, 
                            "weights": weights,
                            "ray_indices": ray_indices,
                            "points_norm": points_norm})
        if self.config.background_model != "none" and self.config.sphere_masking:
            sphere_accum = self.renderer_accumulation(weights=sphere_mask[..., 0],
                                                        ray_indices=ray_indices,
                                                        num_rays=n_rays)
            outputs.update({'sphere_accum': sphere_accum})

        if self.config.background_model != "none":
            if not self.config.sphere_masking:
                bg_field_outputs = self.field_background(bg_ray_samples, compute_uncertainty=False)
            bg_packed_info = nerfacc.pack_info(bg_ray_indices, n_rays)
            bg_weights = nerfacc.render_weight_from_density(
                t_starts=bg_ray_samples.frustums.starts[..., 0],
                t_ends=bg_ray_samples.frustums.ends[..., 0],
                sigmas=bg_field_outputs[FieldHeadNames.DENSITY][..., 0],
                packed_info=bg_packed_info,
            )
            # weights, transmittance, alphas
            bg_weights = bg_weights[0]

            bg_rgb = self.renderer_rgb(rgb=bg_field_outputs[FieldHeadNames.RGB], 
                                ray_indices=bg_ray_indices,
                                num_rays=n_rays,
                                weights=bg_weights)
            bg_depth = self.renderer_depth(weights=bg_weights, 
                                        ray_indices=bg_ray_indices,
                                        num_rays=n_rays,
                                        ray_samples=bg_ray_samples)
            bg_depth = bg_depth / ray_bundle.directions_norm
            bg_accumulation = self.renderer_accumulation(weights=bg_weights,
                                                    ray_indices=bg_ray_indices,
                                                        num_rays=n_rays)
            if self.config.background_model != "none" and self.config.sphere_masking:
                bg_sphere_accum = self.renderer_accumulation(weights=bg_sphere_mask[..., 0],
                                                            ray_indices=bg_ray_indices,
                                                            num_rays=n_rays)
                outputs.update({'bg_sphere_accum':bg_sphere_accum})
            
            # sigma_dup = torch.repeat_interleave(bg_packed_info[:, -1], bg_packed_info[:, -1], 0)
            # sigma_div_up = bg_field_outputs[FieldHeadNames.DENSITY][..., 0]/sigma_dup
            # outputs['mean_sigma'] = torch.unsqueeze(torch.sum(sigma_div_up)/n_rays, dim=0)

            outputs.update({
                'rgb_bg': bg_rgb,
                'accumulation_bg': bg_accumulation,
                'depth_bg': bg_depth,
                "num_samples_per_ray_bg": bg_packed_info[:, 1], # used for creating visiblity mask
                'rgb_inside': rgb,
                'rgb': rgb+bg_rgb*(1-accumulation),
            })
            if self.training:
                outputs.update({                    
                    'weights_bg': bg_weights,
                    "ray_indices_bg": bg_ray_indices,
                })
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["acc_step_size"] = self.sampler.step_size
        return metrics
    
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
                                    bg_render_step_size = self.bg_render_step_size,

                                    alpha_sample_thre=self.config.alpha_sample_thre,
                                    cone_angle=self.config.cone_angle,
                                    grid_sampling=self.config.grid_sampling,
                                    type=self.config.entropy_type,
                                    )

        return topk, mean_uncertainty
    
    def choose_k_views(self, train_set, candidates, topk_list, candidate_cam_info, k=4):
        if len(candidates) <= k:
            maxidx_pose = candidates
        else:
            topk_list = torch.tensor(topk_list)
            maxidx_pose = self.sampler.cal_multiCam(train_set, candidates, topk_list, candidate_cam_info,                                                  
                                                  method=self.config.choose_multi_cam, cand_topk=k)
            # maxidx_pose = torch.topk(self.model.acquisition_grid, self.hparams.topk, dim=0)[1].cpu().numpy()

        return maxidx_pose
    
    def active_iter(self):
        # "random", "frontier", "pixel"
        return self.config.acquisition
