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
Implementation of NeuS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import NeuSSampler
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.nn import Parameter
import pdb
import torch

@dataclass
class UncertNeuSModelConfig(SurfaceModelConfig):
    """NeuS Model Config"""

    _target: Type = field(default_factory=lambda: UncertNeuSModel)
    num_samples: int = 64
    """Number of uniform samples"""
    num_samples_importance: int = 64
    """Number of importance samples"""
    num_up_sample_steps: int = 4
    """number of up sample step, 1 for simple coarse-to-fine sampling"""
    base_variance: float = 64
    """fixed base variance in NeuS sampler, the inv_s will be base * 2 ** iter during upsample"""
    perturb: bool = True
    """use to use perturb for the sampled points"""
    ##############################3
    dist_threshold: float = 0.7
    choose_multi_cam: Literal["dist", "topk"] = "dist"
    ####### base_surface_model modify config
    uncert_rgb_loss_mult: float = 0.01
    """sparse point sdf loss multiplier"""
    uncert_beta_loss_mult: float = 0.01  #0.01~0.25
    """sparse point sdf loss multiplier"""
    uncert_sigma_loss_mult: float = 0.0
    ###### SDF field config
    uncertainty_net: bool = True
    #####################################
    overwrite_near_far_plane: bool = True
    """whether to use near and far collider from command line"""
    near_plane: float = 0.05
    """How far along ray to start sampling."""
    far_plane: float = 1e3
    """How far along ray to stop sampling."""
    scene_contraction_norm: Literal["inf", "l2"] = "inf"
    """Which norm to use for the scene contraction."""


class UncertNeuSModel(SurfaceModel):
    """NeuS model

    Args:
        config: NeuS configuration to instantiate model
    """

    config: UncertNeuSModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        # self.config.pdb_debug = False
        self.sampler = NeuSSampler(
            num_samples=self.config.num_samples,
            num_samples_importance=self.config.num_samples_importance,
            num_samples_outside=self.config.num_samples_outside,
            num_upsample_steps=self.config.num_up_sample_steps,
            base_variance=self.config.base_variance,
            pdb_debug = self.config.pdb_debug,
        )
        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)
        self.lpips = LearnedPerceptualImagePatchSimilarity() #normalize=True)
        self.field = self.config.sdf_field.setup(
            aabb=self.scene_box.aabb,
            spatial_distortion=self.scene_contraction,
            num_images=self.num_train_data,
            uncertainty_net=self.config.uncertainty_net,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )
        self.neus_sampler = self.sampler

        self.anneal_end = 50000

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)
        # anneal for cos in NeuS
        if self.anneal_end > 0:

            def set_anneal(step):
                anneal = min([1.0, step / self.anneal_end])
                self.field.set_cos_anneal_ratio(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )

        return callbacks

    def sample_and_forward_field(self, ray_bundle: RayBundle, steps=None) -> Dict:
        ray_samples = self.neus_sampler(ray_bundle, 
                                        sdf_fn=self.field.get_sdf, steps=steps)
        # save_points("a.ply", ray_samples.frustums.get_start_positions().reshape(-1, 3).detach().cpu().numpy())
        
        # if self.sampler._update_counter.item() <= 0:
        #     field_outputs = self.field(ray_samples, return_alphas=True, 
        #                            return_occupancy=True, 
        #                            steps=steps)
        # else:
        field_outputs = self.field(ray_samples, return_alphas=True, 
                                   return_occupancy=True, 
                                   return_uncertainty=True,
                                   steps=steps)
        if self.config.pdb_debug:
            '''
             field_outputs.keys()
            dict_keys([<FieldHeadNames.RGB: 'rgb'>, <FieldHeadNames.DENSITY: 'density'>, <FieldHeadNames.SDF: 'sdf'>, 
            <FieldHeadNames.NORMAL: 'normal'>, <FieldHeadNames.GRADIENT: 'gradient'>, 'points_norm', 'sampled_sdf',
             <FieldHeadNames.UNCERTAINTY: 'uncertainty'>, <FieldHeadNames.ALPHA: 'alpha'>, <FieldHeadNames.OCCUPANCY: 'occupancy'>])
            '''
            pdb.set_trace()
        if self.config.background_model in ['grid', 'mlp']:
            field_outputs, field_outputs_bg = self.forward_background_field_and_merge(
                ray_samples, field_outputs, return_background=True, steps=steps)
        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs[FieldHeadNames.ALPHA]
        )
        if self.config.pdb_debug:
            pdb.set_trace()
        
        # import pdb
        # pdb.set_trace()
        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            'alpha': field_outputs[FieldHeadNames.ALPHA],
        }
        if self.config.background_model in ['grid', 'mlp']:
            sigma_div_up = field_outputs_bg[FieldHeadNames.DENSITY][..., 0]/ray_samples.shape[1]
            samples_and_field_outputs['mean_sigma'] = torch.unsqueeze(torch.sum(sigma_div_up)/ray_samples.shape[0], dim=0)
        return samples_and_field_outputs
    
    # def get_outputs(self, ray_bundle: RayBundle, steps=None) -> Dict:
    #     samples_and_field_outputs = self.sample_and_forward_field(
    #                                             ray_bundle=ray_bundle,
    #                                             steps=steps)

    #     # Shotscuts
    #     field_outputs = samples_and_field_outputs["field_outputs"]
    #     ray_samples = samples_and_field_outputs["ray_samples"]
    #     weights = samples_and_field_outputs["weights"] # torch.Size([1024, 128, 1])
    #     # import pdb
    #     if self.config.pdb_debug:
    #         pdb.set_trace()

    #     rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
    #     # torch.Size([1024, 3])
    #     depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
    #     # betas = field_outputs[FieldHeadNames.UNCERTAINTY].reshape(weights.shape[0], weights.shape[1], -1)
    #     uncertainty = self.renderer_uncertainty(weights=weights**2, betas = field_outputs[FieldHeadNames.UNCERTAINTY])
    #     alpha = samples_and_field_outputs['alpha']
    #     # the rendered depth is point-to-point distance and we should convert to depth
    #     depth = depth / ray_bundle.directions_norm

    #     # remove the rays that don't intersect with the surface
    #     # hit = (field_outputs[FieldHeadNames.SDF] > 0.0).any(dim=1) & (field_outputs[FieldHeadNames.SDF] < 0).any(dim=1)
    #     # depth[~hit] = 10000.0

    #     normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
    #     accumulation = self.renderer_accumulation(weights=weights)
    #     if self.config.pdb_debug:
    #         # TODO check accumulation always 1?? 
    #         pdb.set_trace()
    #     outputs = {
    #         "rgb": rgb,
    #         "accumulation": accumulation,
    #         "depth": depth,
    #         "normal": normal,
    #         "weights": weights,
    #         # 'alpha': alpha,
    #         "uncertainty" : uncertainty,
    #         "ray_points": self.scene_contraction(
    #             ray_samples.frustums.get_start_positions()
    #         ),  # used for creating visiblity mask
    #         "directions_norm": ray_bundle.directions_norm,  # used to scale z_vals for free space and sdf loss
    #     }

    #     if self.training:
    #         grad_points = field_outputs[FieldHeadNames.GRADIENT]
    #         points_norm = field_outputs["points_norm"]
    #         outputs.update({"eik_grad": grad_points, "points_norm": points_norm})

    #         # TODO volsdf use different point set for eikonal loss
    #         # grad_points = self.field.gradient(eik_points)
    #         # outputs.update({"eik_grad": grad_points})

    #         outputs.update(samples_and_field_outputs)

    #     # this is used only in viewer
    #     outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
    #     if self.config.pdb_debug:
    #         pdb.set_trace()
    #     return outputs

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            # training statics
            metrics_dict["s_val"] = self.field.deviation_network.get_variance().item()
            metrics_dict["inv_s"] = 1.0 / self.field.deviation_network.get_variance().item()

        return metrics_dict
    
    def active_iter(self):
        return "random"
