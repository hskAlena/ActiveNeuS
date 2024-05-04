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

"""Data parser for blendedmvs dataset"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import cv2 as cv
import imageio
import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox

# From sdfstudio_dataparser.py
def get_foreground_masks(image_idx: int, fg_masks):
    """function to process additional foreground_masks

    Args:
        image_idx: specific image index to work with
        fg_masks: foreground_masks
    """

    # sensor depth
    fg_mask = fg_masks[image_idx]

    return {"fg_mask": fg_mask}


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


@dataclass
class BlendedMVSDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: BlendedMVS)
    """target class to instantiate"""
    data: Path = Path("data/BlendedMVS/bmvs_bear")
    """Directory specifying location of data."""
    skip_every_for_val_split: int = 1
    """sub sampling validation images"""
    train_val_no_overlap: bool = False
    """remove selected / sampled validation images from training set"""
    include_foreground_mask: bool = False
    """whether or not to load foreground mask"""
    indices_file: Path = Path("data/BlendedMVS/bmvs_bear/indices.txt")


@dataclass
class BlendedMVS(DataParser):
    """ㅠㅣ Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: BlendedMVSDataParserConfig

    def __init__(self, config: BlendedMVSDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data

    def _generate_dataparser_outputs(self, split="train"):
        camera_dict = np.load(self.data / f"cameras_sphere.npz")
        # scale_mat_xxx.npy
        # world_mat_xxx.npy
        # p_cam = camera_mat @ world_mat @ scale_mat @ p_world

        image_filenames_all = sorted([self.data / "image" / filename for filename in os.listdir(self.data / "image")])
        mask_filenames_all = sorted([self.data / "mask" / filename for filename in os.listdir(self.data / "mask")])
        n_images = len(image_filenames_all)
        indices = list(range(n_images))
        if split != "train" and self.config.skip_every_for_val_split >= 1:
            indices = indices[:: self.config.skip_every_for_val_split]
        else:
            # if you use this option, training set should not contain any image in validation set
            if self.config.train_val_no_overlap:
                indices = [i for i in indices if i % self.config.skip_every_for_val_split != 0]

        indices_file = self.config.indices_file
        indices_file.write_text(json.dumps(indices), "utf8")

        # Filtering based on indices
        image_filenames = [image_filenames_all[i] for i in indices]
        mask_filenames = [mask_filenames_all[i] for i in indices]

        # world_mat is a projection matrix from world to image
        world_mats_np = [camera_dict['world_mat_%d' % i].astype(np.float32) for i in indices]
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        scale_mats_np = [camera_dict['scale_mat_%d' % i].astype(np.float32) for i in indices]

        intrinsics_all = []
        pose_all = []
        fx = []
        fy = []
        cx = []
        cy = []
        foreground_mask_images = []

        for scale_mat, world_mat, mask_filepath in zip(scale_mats_np, world_mats_np, mask_filenames):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            intrinsics = torch.from_numpy(intrinsics).float()
            intrinsics_all.append(intrinsics)
            pose_all.append(torch.from_numpy(pose).float())
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            if self.config.include_foreground_mask:
                # load foreground mask
                foreground_mask = np.array(Image.open(mask_filepath), dtype="uint8")
                foreground_mask = foreground_mask[..., :1]
                foreground_mask_images.append(torch.from_numpy(foreground_mask).float() / 255.0)

        intrinsics_all = torch.stack(intrinsics_all)  # [n_images, 4, 4]
        # intrinsics_all_inv = torch.inverse(intrinsics_all)  # [n_images, 4, 4]

        print(f"{intrinsics_all.shape=}")
        # focal = intrinsics_all[0][0, 0]
        fx = torch.stack(fx)  # 925.54
        fy = torch.stack(fy)  # 922.6
        cx = torch.stack(cx)  # 199.425
        cy = torch.stack(cy)  # 198.103
        # print(f"{focal.shape=}")
        pose_all = torch.stack(pose_all)  # [n_images, 4, 4]
        img_0 = imageio.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]  # (576, 768, 3)

        # poses = np.array(poses).astype(np.float32)

        # img_0 = imageio.imread(image_filenames[0])
        # image_height, image_width = img_0.shape[:2]
        # focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        # cx = image_width / 2.0 #384
        # cy = image_height / 2.0 #288
        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        pose_all[:, 0:3, 1:3] *= -1
        camera_to_world = pose_all[:, :3, :4]  # camera to world transform

        # in x,y,z order
        # camera_to_world[..., 3] *= self.scale_factor
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            height=image_height,
            width=image_width,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,  # <CameraType.PERSPECTIVE: 1>
        )

        additional_inputs_dict = {}
        if self.config.include_foreground_mask:
            additional_inputs_dict["foreground_masks"] = {
                "func": get_foreground_masks,
                "kwargs": {"fg_masks": foreground_mask_images},
            }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames,
            additional_inputs=additional_inputs_dict,
        )

        print(
            f"{len(image_filenames)=} {camera_to_world.shape=} {cx.shape=} {len(mask_filenames)=}, {self.config.include_foreground_mask=}"
        )
        return dataparser_outputs
