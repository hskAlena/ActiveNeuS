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
from typing import Dict, List, Type, Optional
from typing_extensions import Literal

import nerfacc
import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import ActiveNeuSAccSampler
from nerfstudio.models.uncert_neus import UncertNeuSModel, UncertNeuSModelConfig
# cannot import NeuSAcc because of callbacks
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.field_components.encodings import NeRFEncoding
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pdb

@dataclass
class ActiveNeuSAccModelConfig(NeuSModelConfig):
    """UniSurf Model Config"""

    _target: Type = field(default_factory=lambda: ActiveNeuSAccModel)
    sky_loss_mult: float = 0.01
    """Sky segmentation normal consistency loss multiplier."""
    grid_resolution: int = 128
    grid_bg_resolution: int= 64
    steps_warmup: int = 256
    steps_per_grid_update: int = 16
    ##############
    beta_min=0.001
    acquisition: Literal["random", "active", "frontier",'freenerf'] = "random"
    var_eps=1e-6
    entropy_type: Literal[ "ent", "no_surface"] = "ent"
    grid_sampling: bool = True
    kimera_type: Literal[ "none", "active", "entropy"] = "none"
    dist_threshold: float = 1.732
    choose_multi_cam: Literal["dist", "topk"] = "dist"
    ####### base_surface_model modify config
    uncert_rgb_loss_mult: float = 0.001
    """sparse point sdf loss multiplier"""
    uncert_beta_loss_mult: float = 0.01  #0.01~0.25
    uncert_sigma_loss_mult: float = 0.0
    """sparse point sdf loss multiplier"""
    ###### SDF field config
    uncertainty_net: bool = True
    #####################################
    grid_levels: int = 1
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
    alpha_sample_thre: float = 0.0
    """alpha thres for visibility pruning in nerfacc, should set to 0 for the nerf-synthetic dataset"""
    sphere_masking:bool=True
    maintain_aabb: bool=True
    acq_vis_type: Literal["sum", "alpha-composition"] = "sum"




class ActiveNeuSAccModel(NeuSModel):
    """VolSDF model

    Args:
        config: MonoSDF configuration to instantiate model
    """

    config: ActiveNeuSAccModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # voxel surface bybrid sampler from NeuralReconW
        # check diff btw neuralangero and neus_acc
        self.field = self.config.sdf_field.setup(
            aabb=self.scene_box.aabb,
            spatial_distortion=self.scene_contraction,
            num_images=self.num_train_data,
            uncertainty_net=self.config.uncertainty_net,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )
        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)
        self.lpips = LearnedPerceptualImagePatchSimilarity() #normalize=True)
        # self.grid = nerfacc.OccupancyGrid(
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
            self.bg_grid = None
            self.bg_assist_grid = None
            self.bg_aabb = self.scene_aabb
            
        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.bg_render_step_size = None
        else:
            self.bg_render_step_size = self.config.render_step_size
            # self.config.render_step_size = \
            #     ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / 256
            # self.bg_render_step_size = \
            #     ((self.bg_aabb[3:] - self.bg_aabb[:3]) ** 2).sum().sqrt().item() / 256
        
        if self.config.background_model == "mlp":
            position_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=9.0, include_input=True
            )
            direction_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
            )
            self.field_background = NeRFField(
                position_encoding=position_encoding,
                direction_encoding=direction_encoding,
                spatial_distortion=self.scene_contraction,
                use_uncertainty_net=True,
                frequency_regularizer= self.config.sdf_field.frequency_regularizer,
                posenc_len=63, #self.config.sdf_field.posenc_len,
                direnc_len=27,
                freq_reg_end=self.config.sdf_field.freq_reg_end,
                freq_reg_start=self.config.sdf_field.freq_reg_start
            )
        elif self.config.background_model == "grid":
            raise ValueError('Other type of background model not implemented')

        self.background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )
        if self.config.background_model in ['grid', 'mlp']:
            background_model = lambda x, steps: self.field_background.get_density_pts(x, return_uncertainty=True, steps=steps) #* self.config.render_step_size
        else:
            background_model = None
        self.neus_sampler = self.sampler
        self.sampler = ActiveNeuSAccSampler(
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
            sdf_fn=self.field.get_uncertainty,
            inv_s_fn=self.field.deviation_network.get_variance,
            pdb_debug = self.config.pdb_debug
        )

        

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        
        callbacks = super().get_training_callbacks(training_callback_attributes)

        # add sampler call backs
        sdf_uncertainty_fn = self.field.get_uncertainty
        inv_s = self.field.deviation_network.get_variance
        
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.sampler.update_binary_grid,
                # kwargs={"sdf_fn": sdf_uncertainty_fn, "inv_s_fn": inv_s, "bg_model":background_model},
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
        if self.config.pdb_debug:
            pdb.set_trace()
        if self.sampler._update_counter.item() <= 0:
            return super().get_outputs(ray_bundle, steps=steps)
        # self.config.pdb_debug=True
        # import pdb
        # pdb.set_trace()

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
                                        # lower threshold because of uncertainty
                                        alpha_sample_thre=self.config.alpha_sample_thre,
                                        cone_angle=self.config.cone_angle,
                                        bg_model = True
                                    )
        if self.config.pdb_debug:
            pdb.set_trace()
        n_rays = ray_bundle.shape[0]
        field_outputs = self.field(ray_samples, return_alphas=True, return_occupancy=True, 
                                   return_uncertainty=True, steps=steps)
        packed_info = nerfacc.pack_info(ray_indices, n_rays)
        if self.config.background_model != "none" and self.config.sphere_masking:
            field_outputs, bg_field_outputs, sphere_mask, bg_sphere_mask = self.forward_background_field_and_merge_extend(
                                                        bg_ray_samples=bg_ray_samples, 
                                                           ray_samples=ray_samples,
                                                           field_outputs=field_outputs, 
                                                           compute_uncertainty=True, steps=steps,
                                                           return_sphere_mask=True, )
                                                        #    sphere_aabb=1.74*abs(self.scene_box.aabb[0,0]).to(ray_indices.device))
        weights, transmittance = nerfacc.render_weight_from_alpha(
            field_outputs[FieldHeadNames.ALPHA][...,0],
            packed_info=packed_info,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        if self.config.pdb_debug: 
            # SDF all negative??? Nope
            pdb.set_trace()
        # Shotscuts
        # print_alpha = field_outputs[FieldHeadNames.ALPHA]
        # print_uncert = field_outputs[FieldHeadNames.UNCERTAINTY]
        # low_idx = print_alpha < 0.06
        # high_idx = print_uncert > 0.55
        # print_alpha = print_alpha[high_idx]
        # print_uncert = print_uncert[low_idx]
        # print(print_alpha.shape, torch.mean(print_alpha), torch.max(print_alpha), torch.min(print_alpha))
        # print(print_uncert.shape, torch.mean(print_uncert), torch.max(print_uncert), torch.min(print_uncert))
        # input("waiting....")
        sigma_dup = torch.repeat_interleave(packed_info[:, -1], packed_info[:, -1], 0)
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
        uncertainty = self.renderer_uncertainty(weights=weights**2, 
                                                    ray_indices=ray_indices,
                                                    num_rays=n_rays,
                                                    betas=field_outputs[FieldHeadNames.UNCERTAINTY])
        accumulation = self.renderer_accumulation(weights=weights,
                                                    ray_indices=ray_indices,
                                                    num_rays=n_rays)
        ray_points = self.scene_contraction(ray_samples.frustums.get_start_positions())
        # vis_grid = self.assist_grid.occs.reshape(self.assist_grid.binaries.shape)
        
        if self.config.pdb_debug:
            pdb.set_trace()
        outputs={
            "rgb": rgb,
            "accumulation": accumulation,
            # 'alpha': field_outputs[FieldHeadNames.ALPHA],
            "depth": depth,
            "normal": normal,
            'uncertainty': uncertainty,
            "num_samples_per_ray": packed_info[:, 1],
            "ray_points": ray_points,  # used for creating visiblity mask
            "directions_norm": ray_bundle.directions_norm,  # used to scale z_vals for free space and sdf loss
            # 'vis_assist': vis_grid
        }
        # if (steps+1) %self.topk_iter ==0:
        if self.config.acquisition == "frontier":
            inputs = ray_samples.frustums.get_start_positions()
            inputs = inputs.view(-1, 3)
            def make_gauss_entropy(val):
                import numpy as np
                # you can delete 1e-6 when the self.alpha_thres is low enough.
                return torch.log(2*np.pi*val+1e-6)/2 + 1/2

            field_uncertainty = torch.squeeze(field_outputs[FieldHeadNames.UNCERTAINTY], dim=-1)
            neural_implicit_uncert, _ = self.sampler.cal_entropy(inputs, ray_indices, field_uncertainty,
                                        make_gauss_entropy, 'ent', None, self.assist_grid, self.scene_aabb, mean=False )
            neural_implicit_uncert = torch.unsqueeze(neural_implicit_uncert, dim=-1)
            # if self.config.acq_vis_type == "alpha-composition":
            pseudo_weight = weights
            uncertainty_accum = self.renderer_uncertainty(weights=pseudo_weight, 
                                                        ray_indices=ray_indices,
                                                        num_rays=n_rays,
                                                        betas=neural_implicit_uncert)
            uncertainty_accum =  torch.where(uncertainty_accum!=0, uncertainty_accum, torch.log(torch.tensor(1e-6).cuda())/2 + 1/2)
            # else:
            outputs.update({
                "uncertainty_accum_alpha": uncertainty_accum,
            })
            pseudo_weight = torch.ones_like(weights) #/sigma_dup
            uncertainty_accum = self.renderer_uncertainty(weights=pseudo_weight, 
                                                        ray_indices=ray_indices,
                                                        num_rays=n_rays,
                                                        betas=neural_implicit_uncert)
            uncertainty_accum =  torch.where(uncertainty_accum!=0, uncertainty_accum, torch.log(torch.tensor(1e-6).cuda())/2 + 1/2)
            outputs.update({
                "uncertainty_accum": uncertainty_accum,
            })
            
        #################################################################################
        # to visualize active nerf acquisition function
        elif self.config.acquisition == "active" and False:
            ray_samples_neus = self.neus_sampler(ray_bundle, sdf_fn=self.field.get_sdf, steps=steps) # 1024, 128

            ws, betas = self.field.get_alphabeta_nograd(ray_samples_neus, steps=steps)
            weights_neus, transmittance = ray_samples_neus.get_weights_and_transmittance_from_alphas(
                ws # [1024, 128, 1]
            )
            # betas.shape [1024*128, 1]
            beta_shape = betas.reshape(ray_samples_neus.shape[0], -1, 1)
            

            uncertainty_active = self.renderer_uncertainty(weights=weights_neus**2, 
                                                        betas=beta_shape)
            var_pri = beta_shape+ self.config.var_eps
            var_samples = torch.unsqueeze(uncertainty_active, dim=1).repeat(1, ray_samples_neus.shape[1], 1)
            var_post = 1 / (1 / var_pri + ws**2 / (var_samples + self.config.var_eps))
            acq = var_pri - var_post
            # if self.config.acq_vis_type == "alpha-composition":
            pseudo_weight = weights_neus
            uncertainty_active = self.renderer_uncertainty(weights=pseudo_weight, 
                                                        betas=acq)
            outputs.update({
                "uncertainty_active_alpha": uncertainty_active,
            })
            # else:
            uncertainty_active = torch.sum(acq, dim=1)
            outputs.update({
                "uncertainty_active": uncertainty_active,
            })
        ################################################################################
        # if uncertainty.min()>0:
        #     outputs['uncertainty'] = uncertainty
        
        sigma_div_up = field_outputs[FieldHeadNames.ALPHA][..., 0]/sigma_dup
        outputs['mean_sigma'] = torch.unsqueeze(torch.sum(sigma_div_up)/n_rays, dim=0)
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            points_norm = field_outputs["points_norm"]
            outputs.update({"eik_grad": grad_points, 
                            "weights": weights,
                            "ray_samples": ray_samples,
                            "ray_indices": ray_indices,
                            "points_norm": points_norm})
        if self.config.background_model != "none" and self.config.sphere_masking:
            sphere_accum = self.renderer_accumulation(weights=sphere_mask[..., 0],
                                                        ray_indices=ray_indices,
                                                        num_rays=n_rays)
            outputs.update({'sphere_accum': sphere_accum})

        if self.config.background_model != "none":
            if not self.config.sphere_masking:
                bg_field_outputs = self.field_background(bg_ray_samples, compute_uncertainty=True)
            bg_packed_info = nerfacc.pack_info(bg_ray_indices, n_rays)
            bg_weights = nerfacc.render_weight_from_density(
                t_starts=bg_ray_samples.frustums.starts[..., 0],
                t_ends=bg_ray_samples.frustums.ends[..., 0],
                sigmas=bg_field_outputs[FieldHeadNames.DENSITY][..., 0],
                packed_info=bg_packed_info,
            )
            # weights, transmittance, alphas
            bg_weights = bg_weights[0]

            sigma_dup = torch.repeat_interleave(bg_packed_info[:, -1], bg_packed_info[:, -1], 0)
            sigma_div_up = bg_field_outputs[FieldHeadNames.DENSITY][..., 0]/sigma_dup
            outputs['mean_sigma'] = torch.unsqueeze(torch.sum(sigma_div_up)/n_rays, dim=0)
            if self.config.pdb_debug:
                pdb.set_trace()

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
            bg_uncertainty = self.renderer_uncertainty(weights=bg_weights**2, 
                                                    ray_indices=bg_ray_indices,
                                                    num_rays=n_rays,
                                                    betas=bg_field_outputs[FieldHeadNames.UNCERTAINTY])
            if self.config.background_model != "none" and self.config.sphere_masking:
                bg_sphere_accum = self.renderer_accumulation(weights=bg_sphere_mask[..., 0],
                                                            ray_indices=bg_ray_indices,
                                                            num_rays=n_rays)
                outputs.update({'bg_sphere_accum':bg_sphere_accum})
            if self.config.pdb_debug:
                pdb.set_trace()
            outputs.update({
                'rgb_bg': bg_rgb,
                'accumulation_bg': bg_accumulation,
                'uncertainty_bg': bg_uncertainty,
                'depth_bg': bg_depth,
                "num_samples_per_ray_bg": bg_packed_info[:, 1], # used for creating visiblity mask
                'sigma': bg_field_outputs[FieldHeadNames.DENSITY],
                'rgb_inside': rgb,
                'rgb': rgb+bg_rgb*(1-accumulation),
                'uncertainty_inside': uncertainty,
            })
            # if uncertainty.min()>0:
            outputs['uncertainty'] = uncertainty+bg_uncertainty*(1-accumulation)

            if self.training:
                outputs.update({                    
                    'weights_bg': bg_weights,
                    "ray_samples_bg": bg_ray_samples,
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
        
        # vis_grid = self.assist_grid.occs.reshape(self.assist_grid.binaries.shape)
        
        topk, mean_uncertainty = self.sampler.eval_k_views(ray_bundle, 
                                    near_plane=self.config.near_plane,
                                    far_plane=self.config.far_plane,
                                    render_step_size=self.config.render_step_size,
                                    bg_render_step_size = self.bg_render_step_size,
                                    # lower threshold because of uncertainty
                                    alpha_sample_thre=self.config.alpha_sample_thre,
                                    cone_angle=self.config.cone_angle,
                                    grid_sampling=self.config.grid_sampling,
                                    type=self.config.entropy_type,
                                    kimera_type=self.config.kimera_type,
                                    sdf_fn=self.field.get_sdf,)

        if self.config.kimera_type == 'active':
            acq = self.active_eval(ray_bundle)
            topk += acq.item()
            

        return topk, None
    
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
        ray_samples = self.neus_sampler(ray_bundle, sdf_fn=self.field.get_sdf, steps=steps) # 1024, 128

        ws, betas = self.field.get_alphabeta_nograd(ray_samples, steps=steps)
        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
            ws # [1024, 128, 1]
        )
        # betas.shape [1024*128, 1]
        beta_shape = betas.reshape(ray_samples.shape[0], -1, 1)
        

        uncertainty = self.renderer_uncertainty(weights=weights**2, 
                                                    betas=beta_shape)
        var_pri = beta_shape+ self.config.var_eps
        var_samples = torch.unsqueeze(uncertainty, dim=1).repeat(1, ray_samples.shape[1], 1)
        var_post = 1 / (1 / var_pri + ws**2 / (var_samples + self.config.var_eps))
        acq = var_pri - var_post
        acq = torch.unsqueeze(torch.sum(acq), dim=-1).detach().cpu()
        if self.config.kimera_type == 'active':
            acq /= len(var_pri.flatten())
        return acq

    
    def active_iter(self):
        return self.config.acquisition
    
    
