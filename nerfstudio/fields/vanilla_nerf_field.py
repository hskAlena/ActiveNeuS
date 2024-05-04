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

"""Classic NeRF field"""


from typing import Dict, Optional, Tuple, Type

import torch
from torch import nn
from torchtyping import TensorType
from dataclasses import dataclass, field

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    RGBFieldHead,
    UncertaintyFieldHead
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, FieldConfig
from nerfstudio.field_components.encodings import get_freq_reg_mask

# Field related configs
# @dataclass
# class NeRFFieldConfig(FieldConfig):
#     """Configuration for model instantiation"""

#     _target: Type = field(default_factory=lambda: NeRFField)
#     """target class to instantiate"""
    

class NeRFField(Field):
    """NeRF Field

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for ourput head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
        use_integrated_encoding: Used integrated samples as encoding input.
        spatial_distortion: Spatial distortion.
    """
    # config: NeRFFieldConfig

    def __init__(
        self,
        # config: NeRFFieldConfig,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
        field_heads: Tuple[FieldHead] = (RGBFieldHead(),),
        use_integrated_encoding: bool = False,
        use_uncertainty_net: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        frequency_regularizer: bool = False,
        posenc_len: int=63,
        direnc_len: int=27,
        freq_reg_end: int = 14000,
        freq_reg_start: int=0
    ) -> None:
        super().__init__()
        # self.config=config
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.use_integrated_encoding = use_integrated_encoding
        self.spatial_distortion = spatial_distortion
        self.beta_min = 0.001
        self.use_uncertainty_net = use_uncertainty_net
        self.frequency_regularizer = frequency_regularizer
        self.posenc_len = posenc_len
        self.direnc_len = direnc_len
        self.freq_reg_end = freq_reg_end
        self.freq_reg_start = freq_reg_start

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )

        if self.use_uncertainty_net:
            self.uncertainty_net = UncertaintyFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_heads = nn.ModuleList(field_heads)
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore

    def get_density(self, ray_samples: RaySamples, compute_uncertainty:bool=False, steps=None):
        if self.use_integrated_encoding:
            gaussian_samples = ray_samples.frustums.get_gaussian_blob()
            if self.spatial_distortion is not None:
                gaussian_samples = self.spatial_distortion(gaussian_samples)
            encoded_xyz = self.position_encoding(gaussian_samples.mean, covs=gaussian_samples.cov)
        else:
            positions = ray_samples.frustums.get_positions()
            if self.spatial_distortion is not None:
                positions = self.spatial_distortion(positions)
            encoded_xyz = self.position_encoding(positions)

        if self.frequency_regularizer and steps is not None: # and self.training:
            # current_step = kwargs['global_step'] % (self.opt.reload_epoch*1000)
            freq_mask = get_freq_reg_mask(self.posenc_len, steps, self.freq_reg_end, self.freq_reg_start)
            # print(h.shape) ([2097152, 32])
            if encoded_xyz.ndim<=2:
                freq_mask = freq_mask.repeat(encoded_xyz.shape[0],1)
            encoded_xyz = encoded_xyz*freq_mask.to(encoded_xyz.device)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        if compute_uncertainty:
            uncertainty = self.uncertainty_net(base_mlp_out) + self.beta_min
            return density, uncertainty, base_mlp_out
        return density, base_mlp_out
    
    def gradient(self, x, skip_spatial_distortion=False, return_sdf=False, steps=None):
        """compute the gradient of the ray"""
        if self.spatial_distortion is not None and not skip_spatial_distortion:
            x = self.spatial_distortion(x)

        # compute gradient in contracted space
        if self.config.use_numerical_gradients:
            # https://github.com/bennyguo/instant-nsr-pl/blob/main/models/geometry.py#L173
            delta = self.numerical_gradients_delta
            points = torch.stack(
                [
                    x + torch.as_tensor([delta, 0.0, 0.0]).to(x),
                    x + torch.as_tensor([-delta, 0.0, 0.0]).to(x),
                    x + torch.as_tensor([0.0, delta, 0.0]).to(x),
                    x + torch.as_tensor([0.0, -delta, 0.0]).to(x),
                    x + torch.as_tensor([0.0, 0.0, delta]).to(x),
                    x + torch.as_tensor([0.0, 0.0, -delta]).to(x),
                ],
                dim=0,
            )

            points_sdf = self.forward_geonetwork(points.view(-1, 3), 
                                                 steps=steps)[..., 0].view(6, *x.shape[:-1])
            gradients = torch.stack(
                [
                    0.5 * (points_sdf[0] - points_sdf[1]) / delta,
                    0.5 * (points_sdf[2] - points_sdf[3]) / delta,
                    0.5 * (points_sdf[4] - points_sdf[5]) / delta,
                ],
                dim=-1,
            )
        else:
            x.requires_grad_(True)

            y = self.forward_geonetwork(x, steps=steps)[:, :1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
        if not return_sdf:
            return gradients
        else:
            return gradients, points_sdf

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None, steps=None
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)

            if self.frequency_regularizer and steps is not None: # and self.training:
                # current_step = kwargs['global_step'] % (self.opt.reload_epoch*1000)
                freq_mask = get_freq_reg_mask(self.direnc_len, steps, self.freq_reg_end, self.freq_reg_start)
                # print(h.shape) ([2097152, 32])
                if encoded_dir.ndim<=2:
                    freq_mask = freq_mask.repeat(encoded_dir.shape[0],1)
                encoded_dir = encoded_dir*freq_mask.to(encoded_dir.device)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs
    
    def get_density_pts(self, positions, return_feature=False, return_uncertainty=False, steps=None):
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
        encoded_xyz = self.position_encoding(positions)
        if self.frequency_regularizer and steps is not None: # and self.training:
            # current_step = kwargs['global_step'] % (self.opt.reload_epoch*1000)
            freq_mask = get_freq_reg_mask(self.posenc_len, steps, self.freq_reg_end, self.freq_reg_start)
            # print(h.shape) ([2097152, 32])
            freq_mask = freq_mask.repeat(encoded_xyz.shape[0],1)
            encoded_xyz = encoded_xyz*freq_mask.to(encoded_xyz.device)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        if return_uncertainty:
            uncertainty = self.uncertainty_net(base_mlp_out) + self.beta_min
            return density, uncertainty, base_mlp_out
        if return_feature:
            return density, base_mlp_out
        return density
