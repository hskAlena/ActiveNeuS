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
Semantic datamanager.
"""

from dataclasses import dataclass, field
from typing import Type, Dict, Tuple

from nerfstudio.data.utils.dataloaders import (
    FixedIndicesDataloader,
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from rich.progress import Console
CONSOLE = Console(width=120)
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datasets.semantic_dataset import SemanticDataset


@dataclass
class ActiveDataManagerConfig(VanillaDataManagerConfig):
    """A semantic datamanager - required to use with .setup()"""

    _target: Type = field(default_factory=lambda: ActiveDataManager)
    # uncertainty_net: bool = True
    num_topk_images: int = 4

class ActiveDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """Data manager implementation for data that also requires processing semantic data.

    Args:
        config: the DataManagerConfig used to instantiate class
    """
   
    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = FixedIndicesDataloader(
            self.train_dataset,
            # num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_topk_images=self.config.num_topk_images,
            device=self.device,
            num_workers=self.world_size * 2,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            dataset_config = self.config
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, 
                                                           self.config.train_num_rays_per_batch,
                                                           precrop_iters=self.config.precrop_iters,
                                                           precrop_frac=self.config.precrop_frac)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        if self.test_mode != 'train':
            self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
            precrop_iters=0,
            precrop_frac=self.config.precrop_frac,
            )
        else:
            self.train_ray_generator = RayGenerator(
                self.train_dataset.cameras.to(self.device),
                self.train_camera_optimizer,
                precrop_iters=self.config.precrop_iters,
                precrop_frac=self.config.precrop_frac,
            )
        self.train_candidate = list(self.train_image_dataloader.train_candidate)
        self.i_train = list(self.train_image_dataloader.i_train)
        self.cameras = self.train_image_dataloader.cameras

        # for loading full images in extract_mesh
        self.fixed_indices_train_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 2,
            shuffle=False,
            dataset_config = self.config
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        # image_idx, image (49, 384, 384, 3)
        batch = self.train_pixel_sampler.sample(image_batch, steps=step)
        # image (1024, 3), indices 
        """
        'indices': tensor([[  9, 150, 310],
        [  8, 294, 349],
        [ 19, 153,  68],
        ...,
        [  0, 324,   7],
        [  8,  22, 343],
        [ 40,  94,  26]])}
        """
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices, steps=step)
        # origins, directions, pixel_area, directions_norm, camera_indices,
        # nears, fars, metadata, times
        return ray_bundle, batch
    
    def train_candidate_iter(self, idx):
        cand_idx = self.train_candidate[idx]
        image_batch = self.train_image_dataloader._get_collated_idx(cand_idx)
        # import pdb
        # pdb.set_trace()
        batch = self.train_pixel_sampler.sample(image_batch)

        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch
    
    def train_candidate_caminfo(self):
        return_dict = {}
        poses = self.cameras.camera_to_worlds
        img_w = self.cameras.width
        img_h = self.cameras.height
        K = (self.cameras.fx, self.cameras.fy, 
             self.cameras.cx, self.cameras.cy)

        return_dict['poses'] = poses
        return_dict['img_w_h'] = (img_w, img_h)
        return_dict['fx_fy_cx_cy'] = K
        """
        camera_to_worlds (B, 3,4)
        fx, fy, cx, cy, width, height, distortion_params, camera_type, times
        """
        return return_dict
    
    def train_candidate_cam_iter(self, idx):
        cand_idx = self.train_candidate[idx]
        ray_bundle, batch = self.train_image_dataloader.get_data_from_image_idx(cand_idx)
        return ray_bundle, batch
    
    def update_bundle(self, topk_add):
        self.train_candidate = list(self.train_image_dataloader.update_candidate(topk_add))
        self.i_train = list(self.train_image_dataloader.i_train)
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)