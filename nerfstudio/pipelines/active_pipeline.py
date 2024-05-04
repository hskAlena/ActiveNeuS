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
A pipeline that dynamically chooses the number of rays to sample.
"""

from dataclasses import dataclass, field
from typing import Type

import torch
from scipy.spatial import distance
from typing_extensions import Literal

from nerfstudio.data.datamanagers.active_datamanager import ActiveDataManager
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
import random
import numpy as np
from tqdm import tqdm
import time
import typing
from nerfstudio.models.base_model import Model, ModelConfig
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist



@dataclass
class ActivePipelineConfig(VanillaPipelineConfig):
    """Dynamic Batch Pipeline Config"""

    _target: Type = field(default_factory=lambda: ActivePipeline)
    topk_iter: int = 4000
    hardcode_candidate: str =None


class ActivePipeline(VanillaPipeline):
    """Pipeline with logic for changing the number of rays per batch."""

    # pylint: disable=abstract-method

    config: ActivePipelineConfig
    datamanager: ActiveDataManager
    dynamic_num_rays_per_batch: int

    def __init__(
            self,
            config: ActivePipelineConfig,
            device: str,
            test_mode: Literal["test", "val", "inference"] = "val",
            world_size: int = 1,
            local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        assert isinstance(
            self.datamanager, ActiveDataManager
        ), "ActivePipeline only works with ActiveDataManager."
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            topk_iter = self.config.topk_iter,
            metadata=self.datamanager.train_dataset.metadata,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    def get_train_loss_dict(self, step: int):

        ray_bundle, batch = self.datamanager.next_train(step)
        if step==0 or (step-1)%self.config.topk_iter ==0:
            datapath = str(self.datamanager.dataparser.config.data.stem)
            # print(datapath)
            if 'scan' in datapath:
                # DTU dataset
                self._model.initialize_grid(self.datamanager.train_dataset.cameras, self.datamanager.i_train, transform=True)
            else:
                self._model.initialize_grid(self.datamanager.train_dataset.cameras, self.datamanager.i_train, transform=True)
        model_outputs = self._model(ray_bundle, steps=step)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
        if camera_opt_param_group in self.datamanager.get_param_groups():
            # Report the camera optimization metrics
            metrics_dict["camera_opt_translation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
            )
            metrics_dict["camera_opt_rotation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
            )

        if 'depth' in batch and 'inv_s' in metrics_dict:
            cam_info = self.datamanager.train_candidate_caminfo()
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict, cam_info)
        else:
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        if step>0 and step % self.config.topk_iter == 0:
            topk_list = []
            candidates = self.datamanager.train_candidate
            i_train = self.datamanager.i_train
            if len(i_train) >= self.datamanager.config.num_topk_images*5:
                return model_outputs, loss_dict, metrics_dict
            candidate_cam_info = self.datamanager.train_candidate_caminfo()
            select_scheme = self._model.active_iter()
            s = time.time()
            if select_scheme == 'frontier' or select_scheme=='entropy':
                if len(candidates) > self.datamanager.config.num_topk_images:
                    for i in tqdm(range(len(candidates))):
                        ray_bundle, batch = self.datamanager.train_candidate_iter(i)
                        cand_acq_value, vis_assist = self._model.eval_k_views(ray_bundle)
                        # print(mean_uncertainty)
                        topk_list.append(cand_acq_value)
                    # model_outputs['vis_assist'] = vis_assist
                    topk_add = self._model.choose_k_views(i_train, candidates, topk_list, candidate_cam_info, 
                                                        k=self.datamanager.config.num_topk_images)
                else:
                    topk_add = candidates
            elif select_scheme == 'random':
                # topk_add = torch.random(candidates, self.datamanager.config.num_topk_images)
                if len(candidates) > self.datamanager.config.num_topk_images:
                    tmp_candidate = set(candidates)
                    topk_add = np.array(random.sample(tmp_candidate, self.datamanager.config.num_topk_images))
                else:
                    topk_add = candidates
            elif select_scheme == 'active':
                if len(candidates) > self.datamanager.config.num_topk_images:
                    with torch.no_grad():
                        for i in tqdm(range(len(candidates))):
                            ray_bundle, batch = self.datamanager.train_candidate_iter(i)
                            topk_list.append(self._model.active_eval(ray_bundle, steps=step))
                    # topk_list = torch.cat(topk_list, 0)
                    # topk_add = torch.topk(topk_list, self.datamanager.config.num_topk_images)[1].cpu().numpy()
                    # candidates = np.array(candidates)
                    # topk_add = candidates[topk_add.astype(int)]
                    topk_add = self._model.choose_k_views(i_train, candidates, topk_list, candidate_cam_info, 
                                                        k=self.datamanager.config.num_topk_images)
                else:
                    topk_add = candidates
            elif select_scheme == 'fvs':
                if len(candidates) > self.datamanager.config.num_topk_images:
                    poses = candidate_cam_info['poses'].detach().cpu().numpy()  # [100, 3, 4]
                    # poses: Matrices
                    # candidates: Indices from original set
                    # i_train: Indices from original set
                    topk_add =np.array([], dtype=np.int32)
                    i_train = np.array(i_train, dtype=np.int32)
                    cand_ids = np.array(candidates)
                    for j in range(self.datamanager.config.num_topk_images):
                        train_ids = np.concatenate((i_train, topk_add), 0, dtype=np.int32)
                        if j>0:
                            cand_ids = np.delete(cand_ids, np.where(cand_ids == topk_add[j-1]))
                        train_poses_xyz = np.squeeze(poses[cand_ids][..., 3:], -1)  # not used
                        i_train_poses_xyz = np.squeeze(poses[train_ids][..., 3:], -1)  # already used
                        # Compute pairwise distances among candidates
                        pairwise_distances = distance.cdist(train_poses_xyz, i_train_poses_xyz, metric='euclidean')

                        # Get the indices of the furthest candidates
                        sum_distances = np.sum(pairwise_distances, axis=1)
                        sorted_indices = np.argsort(sum_distances)[::-1]  # sort in descending order
                        sorted_train_poses = train_poses_xyz[sorted_indices]          
                        topk_add = np.append(topk_add, np.array(candidates)[sorted_indices[0]])
                else:
                    topk_add = candidates
            elif select_scheme=='freenerf':
                datapath = str(self.datamanager.dataparser.config.data)
                if 'dtu' in datapath:
                    init_ = [25, 22, 28, 40, 44, 48, 0, 8, 13]
                    if self.config.hardcode_candidate is not None:
                        init_ = eval(self.config.hardcode_candidate)
                elif 'blender' in datapath:
                    init_ = [26, 86, 2, 55, 75, 93, 16, 73, 8]
                    if 'chair' in datapath:
                        init_ = [26, 86, 28, 58, 18, 24, 95, 11, 13, 74]
                    elif 'drums' in datapath:
                        init_ = [26, 86, 29,  0, 61, 82, 57, 14, 74, 17]
                    elif 'materials' in datapath:
                        init_ = [26, 86, 47, 25, 79, 91, 63, 30, 32, 40]
                    elif 'lego' in datapath:
                        init_ = [26, 86, 8,  4, 90, 50, 35, 37, 49, 59]
                    elif 'hotdog' in datapath:
                        init_ = [26, 86, 54, 36, 97, 23, 34, 41, 87, 16]
                    elif 'mic' in datapath:
                        init_ = [26, 86, 49, 28, 72,  7, 90, 69,  3, 91]
                    elif 'ficus' in datapath:
                        init_ = [26, 86, 12,  0, 74, 75,  4, 80, 60,  6]
                    elif 'ship' in datapath:
                        init_ = [26, 86, 78, 58, 62, 39, 27,  2, 45, 38]
                    # if 'chair' in datapath:
                    #     init_ = [26, 86, 2, 55, 81, 49, 60, 40, 48, 6, 59, 70, 97, 52,  4, 24, 0, 14, 43, 85]
                    # elif 'drums' in datapath:
                    #     init_ = [26, 86, 2, 55, 42, 65, 80, 97, 61, 85, 56, 58, 37, 50, 54, 31, 45, 74, 70, 20]
                    # elif 'materials' in datapath:
                    #     init_ = [26, 86, 2, 55, 82, 60, 59, 81,  1, 97, 67, 91, 79, 46, 94, 50, 71, 25, 72, 14]
                    # elif 'lego' in datapath:
                    #     init_ = [26, 86, 2, 55, 30, 92, 81, 57, 67, 73, 97,  4, 77, 17, 25, 58,  0, 88, 76, 96]
                    # elif 'hotdog' in datapath:
                    #     init_ = [26, 86, 2, 55, 32, 39, 69, 34, 87, 64, 72, 25, 97, 18, 38,  1,  5, 68, 49, 42]
                    # elif 'mic' in datapath:
                    #     init_ = [26, 86, 2, 55, 45, 94, 92, 96,  5, 58, 59, 35, 15, 48, 28, 33, 39, 44, 75, 60]
                    # elif 'ficus' in datapath:
                    #     init_ = [26, 86, 2, 55, 65, 46,  1, 38, 47, 25,  9,  4, 56, 31, 70, 42, 90, 32, 94, 60]
                    # elif 'ship' in datapath:
                    #     init_ = [26, 86, 2, 55, 90, 32, 78, 57, 27, 38, 63,  1, 45, 41, 52, 85, 83, 36, 16, 96]
                final_flag = len(i_train)+self.datamanager.config.num_topk_images
                if len(i_train)>=len(init_):
                    topk_add = np.array([])
                elif final_flag > len(init_):
                    final_flag = len(init_)
                    topk_add = np.array(init_[len(i_train):final_flag])
                else:
                    topk_add = np.array(init_[len(i_train):final_flag])
                
            e = time.time()
            print(select_scheme, "Time passed", e-s)
            metrics_dict['select_time_duration'] = e-s
            for i in range(len(topk_add)):
                metrics_dict[f'select_topk_{i}_add'] = topk_add[i]
            # self.log('select/time', e-s)
            self.datamanager.update_bundle(topk_add)
            print("NOTICE: train set updated : ", topk_add)

        return model_outputs, loss_dict, metrics_dict

    def get_eval_loss_dict(self, step: int):
        model_outputs, loss_dict, metrics_dict = super().get_eval_loss_dict(step)

        # add the number of rays
        assert "num_rays_per_batch" not in metrics_dict
        metrics_dict["num_rays_per_batch"] = torch.tensor(self.datamanager.eval_pixel_sampler.num_rays_per_batch)

        return model_outputs, loss_dict, metrics_dict
