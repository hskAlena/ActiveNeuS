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
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import (
    Config,
    SchedulerConfig,
    TrainerConfig,
    ViewerConfig,
)
from nerfstudio.data.datamanagers.base_datamanager import (
    FlexibleDataManagerConfig,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datamanagers.active_datamanager import ActiveDataManagerConfig

from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig, AdamWOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
    NeuSSchedulerConfig,
    MultiStepWarmupSchedulerConfig,
)
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.sdf_field import SDFFieldConfig
# from nerfstudio.fields.vanilla_nerf_field import NeRFFieldConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.neuralangelo import NeuralangeloModelConfig
from nerfstudio.models.neuralangelo_acc import NeuralangeloAccModelConfig
from nerfstudio.models.neuralangelo_uncert_acc import NeuralangeloUncertAccModelConfig

# from nerfstudio.models.neuralreconW import NeuralReconWModelConfig
from nerfstudio.models.neus import NeuSModelConfig
from nerfstudio.models.neus_acc import NeuSAccModelConfig
from nerfstudio.models.active_neus_acc import ActiveNeuSAccModelConfig
from nerfstudio.models.uncert_neus import UncertNeuSModelConfig

from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.models.nerf_acc import NeRFAccModel, NeRFAccModelConfig
from nerfstudio.pipelines.base_pipeline import (
    FlexibleInputPipelineConfig,
    VanillaPipelineConfig,
)
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.pipelines.active_pipeline import ActivePipelineConfig
from nerfstudio.pipelines.dynamic_batch_active import DynamicBatchActivePipelineConfig

method_configs: Dict[str, Config] = {}
descriptions = {
    "active-neus": "Implementation of NeuS.",
    "uncert-neus": "Implementation of NeuS.",
    "neus": "Implementation of NeuS.",

    "neus-acc": "Implementation of NeuS with empty space skipping.",
    "active_neus-acc": "Implementation of NeuS in active learning with empty space skipping.",
    "active_uncert_neus-acc": "Implementation of Uncertainty NeuS in active learning with empty space skipping.",
    "uncert_neus-acc": "Implementation of Uncertainty NeuS with empty space skipping.",

    "vanilla-nerf": "Original NeRF model. (slow)",
}

method_configs["active_uncert_neus-acc"] = Config(
    method_name="active_uncert_neus-acc",
    trainer=TrainerConfig(
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20001,
        mixed_precision=False,
    ),
    pipeline=ActivePipelineConfig(
        datamanager=ActiveDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=1024, #2048,
            eval_num_rays_per_batch=512, #1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        ## TODO check sdf field conf, use_grid_feature True
        model=ActiveNeuSAccModelConfig(eval_num_rays_per_chunk=256), # 1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15, weight_decay=0.0),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15, weight_decay=0.0),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["active_neus-acc"] = Config(
    method_name="active_neus-acc",
    trainer=TrainerConfig(
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=4000, #20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20001,
        mixed_precision=False,
    ),
    pipeline=ActivePipelineConfig(
        datamanager=ActiveDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=1024, #2048,
            eval_num_rays_per_batch=256, #1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        ## TODO check sdf field conf, use_grid_feature True
        model=NeuSAccModelConfig(eval_num_rays_per_chunk=256), # 1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15, weight_decay=0.0),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15, weight_decay=0.0),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["neus"] = Config(
    method_name="neus",
    trainer=TrainerConfig(
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSModelConfig(eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["neus-acc"] = Config(
    method_name="neus-acc",
    trainer=TrainerConfig(
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=512, #2048,
            eval_num_rays_per_batch=512, #1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSAccModelConfig(eval_num_rays_per_chunk=512), #1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15, weight_decay=0.0),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15, weight_decay=0.0),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["active-nerf-acc"] = Config(
    method_name="active-nerf-acc",
    pipeline=ActivePipelineConfig(
        datamanager=ActiveDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=NeRFAccModelConfig(_target=NeRFAccModel),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15, weight_decay=0.0),
            "scheduler": ExponentialDecaySchedulerConfig(warmup_steps=0, lr_final=0.0001, max_steps=20000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["nerf-acc"] = Config(
    method_name="nerf-acc",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=NeRFAccModelConfig(_target=NeRFAccModel),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15, weight_decay=0.0),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=20000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["vanilla-nerf"] = Config(
    method_name="vanilla-nerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=VanillaModelConfig(_target=NeRFModel),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    },
)


AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
