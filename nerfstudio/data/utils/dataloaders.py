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
Code for sampling images from a dataset of images.
"""

# for multithreading
import concurrent.futures
import multiprocessing
import random
from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
import numpy as np
from rich.progress import Console, track
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.utils.misc import get_dict_to_torch

CONSOLE = Console(width=120)


class CacheDataloader(DataLoader):
    """Collated image dataset that implements caching of default-pytorch-collatable data.
    Creates batches of the InputDataset return type.

    Args:
        dataset: Dataset to sample from.
        num_samples_to_collate: How many images to sample rays for each batch. -1 for all images.
        num_times_to_repeat_images: How often to collate new images. -1 to never pick new images.
        device: Device to perform computation.
        collate_fn: The function we will use to collate our training data
    """

    def __init__(
        self,
        dataset: Dataset,
        num_images_to_sample_from: int = -1,
        num_times_to_repeat_images: int = -1,
        device: Union[torch.device, str] = "cpu",
        collate_fn=nerfstudio_collate,
        **kwargs,
    ):
        self.dataset = dataset
        super().__init__(dataset=dataset, **kwargs)  # This will set self.dataset
        self.num_times_to_repeat_images = num_times_to_repeat_images
        self.cache_all_images = (num_images_to_sample_from == -1) or (num_images_to_sample_from >= len(self.dataset))
        self.num_images_to_sample_from = len(self.dataset) if self.cache_all_images else num_images_to_sample_from
        self.device = device
        self.collate_fn = collate_fn
        self.num_workers = kwargs.get("num_workers", 0)

        self.num_repeated = self.num_times_to_repeat_images  # starting value
        self.first_time = True

        self.cached_collated_batch = None
        if self.cache_all_images:
            CONSOLE.print(f"Caching all {len(self.dataset)} images.")
            if len(self.dataset) > 500:
                CONSOLE.print(
                    "[bold yellow]Warning: If you run out of memory, try reducing the number of images to sample from."
                )
            self.cached_collated_batch = self._get_collated_batch()
        elif self.num_times_to_repeat_images == -1:
            CONSOLE.print(
                f"Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, without resampling."
            )
        else:
            CONSOLE.print(
                f"Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, "
                f"resampling every {self.num_times_to_repeat_images} iters."
            )

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""

        indices = random.sample(range(len(self.dataset)), k=self.num_images_to_sample_from)
        batch_list = []
        results = []

        num_threads = int(self.num_workers) * 4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)

            for res in track(
                results, description="Loading data batch", transient=True, disable=(self.num_images_to_sample_from == 1)
            ):
                batch_list.append(res.result())

        return batch_list

    def _get_collated_batch(self):
        """Returns a collated batch."""
        batch_list = self._get_batch_list()
        collated_batch = self.collate_fn(batch_list)
        collated_batch = get_dict_to_torch(collated_batch, device=self.device, exclude=["image"])
        return collated_batch

    def __iter__(self):
        while True:
            if self.cache_all_images:
                collated_batch = self.cached_collated_batch
            elif self.first_time or (
                self.num_times_to_repeat_images != -1 and self.num_repeated >= self.num_times_to_repeat_images
            ):
                # trigger a reset
                self.num_repeated = 0
                collated_batch = self._get_collated_batch()
                # possibly save a cached item
                self.cached_collated_batch = collated_batch if self.num_times_to_repeat_images != 0 else None
                self.first_time = False
            else:
                collated_batch = self.cached_collated_batch
                self.num_repeated += 1
            yield collated_batch

class FixedIndicesDataloader(DataLoader):
    """Collated image dataset that implements caching of default-pytorch-collatable data.
    Creates batches of the InputDataset return type.

    Args:
        dataset: Dataset to sample from.
        num_samples_to_collate: How many images to sample rays for each batch. -1 for all images.
        num_times_to_repeat_images: How often to collate new images. -1 to never pick new images.
        device: Device to perform computation.
        collate_fn: The function we will use to collate our training data
    """

    def __init__(
        self,
        dataset: Dataset,
        num_topk_images: int = 4,
        device: Union[torch.device, str] = "cpu",
        collate_fn=nerfstudio_collate,
        **kwargs,
    ):
        self.dataset = dataset
        datapath = str(kwargs['dataset_config'].dataparser.data)
        # 'data/dtu/scan55'
        del kwargs['dataset_config']
        super().__init__(dataset=dataset, **kwargs)  # This will set self.dataset
        # self.cache_all_images = (num_images_to_sample_from == -1) or (num_images_to_sample_from >= len(self.dataset))
        # self.num_images_to_sample_from = len(self.dataset) if self.cache_all_images else num_images_to_sample_from
        self.cameras = dataset.cameras.to(device)
        self.device = device
        self.collate_fn = collate_fn
        self.num_workers = kwargs.get("num_workers", 0)

        self.first_time = False
        
        if 'dtu' in datapath:
            init_ = [25, 22, 28, 40, 44, 48, 0, 8, 13]
            eval_ = [1, 9, 12, 15, 24, 27, 32, 35, 42, 46]
        elif 'blender' in datapath:
            init_ = [26, 86, 2, 55, 75, 93, 16, 73, 8]
            if num_topk_images == 10:
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
            elif num_topk_images == 20:
                if 'chair' in datapath:
                    init_ = [26, 86, 2, 55, 81, 49, 60, 40, 48, 6, 59, 70, 97, 52,  4, 24, 0, 14, 43, 85]
                elif 'drums' in datapath:
                    init_ = [26, 86, 2, 55, 42, 65, 80, 97, 61, 85, 56, 58, 37, 50, 54, 31, 45, 74, 70, 20]
                elif 'materials' in datapath:
                    init_ = [26, 86, 2, 55, 82, 60, 59, 81,  1, 97, 67, 91, 79, 46, 94, 50, 71, 25, 72, 14]
                elif 'lego' in datapath:
                    init_ = [26, 86, 2, 55, 30, 92, 81, 57, 67, 73, 97,  4, 77, 17, 25, 58,  0, 88, 76, 96]
                elif 'hotdog' in datapath:
                    init_ = [26, 86, 2, 55, 32, 39, 69, 34, 87, 64, 72, 25, 97, 18, 38,  1,  5, 68, 49, 42]
                elif 'mic' in datapath:
                    init_ = [26, 86, 2, 55, 45, 94, 92, 96,  5, 58, 59, 35, 15, 48, 28, 33, 39, 44, 75, 60]
                elif 'ficus' in datapath:
                    init_ = [26, 86, 2, 55, 65, 46,  1, 38, 47, 25,  9,  4, 56, 31, 70, 42, 90, 32, 94, 60]
                elif 'ship' in datapath:
                    init_ = [26, 86, 2, 55, 90, 32, 78, 57, 27, 38, 63,  1, 45, 41, 52, 85, 83, 36, 16, 96]
        i_train = init_[:num_topk_images]
        self.i_train = i_train
        self.train_candidate = list(np.arange(len(self.dataset))) #set(range(len(self.dataset))) 
        if 'dtu' in datapath:
            # self.train_candidate -= set(eval_)
            # self.train_candidate = np.delete(self.train_candidate, self.train_candidate==np.array(eval_))
            self.train_candidate = [cand for cand in self.train_candidate if cand not in eval_ ]
        # self.train_candidate -=self.i_train
        self.train_candidate = [cand for cand in self.train_candidate if cand not in self.i_train ]

        # self.cached_collated_batch = None
        # if self.cache_all_images:
        CONSOLE.print(f"Caching all {len(self.i_train)} images.")
        # if len(self.dataset) > 500:
        #     CONSOLE.print(
        #         "[bold yellow]Warning: If you run out of memory, try reducing the number of images to sample from."
        #     )
        self.cached_collated_batch = self._get_collated_batch()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""
        print("DATALOADER: i_train : ", self.i_train)

        indices = random.sample(self.i_train, k=len(self.i_train))
        batch_list = []
        results = []

        num_threads = int(self.num_workers) * 4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)

            for res in track(
                results, description="Loading data batch", transient=True, disable=False
            ):
                batch_list.append(res.result())

        return batch_list

    def update_candidate(self, topk_add):
        self.i_train += list(topk_add)
        # topk_add = set(topk_add)
        # self.train_candidate -= topk_add
        self.train_candidate = [cand for cand in self.train_candidate if cand not in topk_add ]
        self.first_time = True
        return self.train_candidate
    
    def get_data_from_image_idx(self, image_idx: int) -> Tuple[RayBundle, Dict]:
        """Returns the data for a specific image index.

        Args:
            image_idx: Camera image index
        """
        # import pdb
        # pdb.set_trace()
        ray_bundle = self.cameras.generate_rays(camera_indices=image_idx, keep_shape=True)
        batch = self.dataset[image_idx]
        new_batch = dict()
        new_batch['image_idx'] = torch.tensor([batch['image_idx']], dtype=torch.uint8)
        new_batch['image'] = torch.unsqueeze(batch['image'], dim=0)
        new_batch = get_dict_to_torch(new_batch, device=self.device, exclude=["image"])
        return ray_bundle, new_batch
    
    def _get_collated_idx(self, idx):
        """Returns a collated batch."""
        # import pdb
        # pdb.set_trace()
        batch_list = self.dataset[idx]
        new_batch = dict()
        new_batch['image_idx'] = torch.tensor([batch_list['image_idx']], dtype=torch.uint8)
        new_batch['image'] = torch.unsqueeze(batch_list['image'], dim=0)
        collated_batch = get_dict_to_torch(new_batch, device=self.device, exclude=["image"])
        return collated_batch
        
    def _get_collated_batch(self):
        """Returns a collated batch."""
        batch_list = self._get_batch_list()
        #image_idx, image
        collated_batch = self.collate_fn(batch_list)
        collated_batch = get_dict_to_torch(collated_batch, device=self.device, exclude=["image"])
        return collated_batch

    def __iter__(self):
        while True:
            if self.first_time:                # trigger a reset
                collated_batch = self._get_collated_batch()
                # possibly save a cached item
                self.cached_collated_batch = collated_batch
                self.first_time = False
            else:
                collated_batch = self.cached_collated_batch
            yield collated_batch

class EvalDataloader(DataLoader):
    """Evaluation dataloader base class

    Args:
        input_dataset: InputDataset to load data from
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        self.input_dataset = input_dataset
        self.cameras = input_dataset.cameras.to(device)
        self.device = device
        self.kwargs = kwargs
        super().__init__(dataset=input_dataset)

    @abstractmethod
    def __iter__(self):
        """Iterates over the dataset"""
        return self

    @abstractmethod
    def __next__(self) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data"""

    def get_camera(self, image_idx: int = 0) -> Cameras:
        """Get camera for the given image index

        Args:
            image_idx: Camera image index
        """
        return self.cameras[image_idx]

    def get_data_from_image_idx(self, image_idx: int) -> Tuple[RayBundle, Dict]:
        """Returns the data for a specific image index.

        Args:
            image_idx: Camera image index
        """
        ray_bundle = self.cameras.generate_rays(camera_indices=image_idx, keep_shape=True)
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        return ray_bundle, batch


class FixedIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns a fixed set of indices.

    Args:
        input_dataset: InputDataset to load data from
        image_indices: List of image indices to load data from. If None, then use all images.
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        image_indices: Optional[Tuple[int]] = None,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(input_dataset, device, **kwargs)
        if image_indices is None:
            self.image_indices = list(range(len(input_dataset)))
        else:
            self.image_indices = image_indices
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self.image_indices):
            image_idx = self.image_indices[self.count]
            ray_bundle, batch = self.get_data_from_image_idx(image_idx)
            self.count += 1
            return ray_bundle, batch
        raise StopIteration


class RandIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns random images.

    Args:
        input_dataset: InputDataset to load data from
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(input_dataset, device, **kwargs)
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < 1:
            image_indices = range(self.cameras.size)
            image_idx = random.choice(image_indices)
            ray_bundle, batch = self.get_data_from_image_idx(image_idx)
            self.count += 1
            return ray_bundle, batch
        raise StopIteration
