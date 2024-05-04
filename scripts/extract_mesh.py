#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import Tuple

import torch
import tyro
from rich.console import Console

from nerfstudio.model_components.ray_samplers import save_points
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.marching_cubes import (
    get_surface_occupancy,
    get_surface_sliding,
    get_surface_sliding_with_contraction,
)

CONSOLE = Console(width=120)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


@dataclass
class ExtractMesh:
    """Load a checkpoint, run marching cubes, extract mesh, and save it to a ply file."""

    # Path to config YAML file.
    load_config: Path
    # Marching cube resolution.
    resolution: int = 1024
    # Name of the output file.
    output_path: Path = Path("output.ply")
    # Whether to simplify the mesh.
    simplify_mesh: bool = False
    # extract the mesh using occupancy field (unisurf) or SDF, default sdf
    is_occupancy: bool = False
    """Minimum of the bounding box."""
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    """Maximum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """marching cube threshold"""
    marching_cube_threshold: float = 0.0
    """create visibility mask"""
    create_visibility_mask: bool = False
    """save visibility grid"""
    save_visibility_grid: bool = False
    """visibility grid resolution"""
    visibility_grid_resolution: int = 512
    """threshold for considering a points is valid when splat to visibility grid"""
    valid_points_thres: float = 0.005
    """sub samples factor of images when creating visibility grid"""
    sub_sample_factor: int = 8
    """torch precision"""
    torch_precision: Literal["highest", "high"] = "high"
    mesh_list: str = None
    
    def main(self) -> None:
        """Main function."""
        torch.set_float32_matmul_precision(self.torch_precision)
        assert str(self.output_path)[-4:] == ".ply"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        config, pipeline, _ = eval_setup(self.load_config)

        eval_step = 200000
        if 'neus' in config.method_name:
            pipeline.model.sampler.update_step_size(eval_step, pipeline.model.field.deviation_network.get_variance)

        CONSOLE.print("Extract mesh with marching cubes and may take a while")

        if self.create_visibility_mask:
            assert self.resolution % 512 == 0

            coarse_mask = pipeline.get_visibility_mask(
                self.visibility_grid_resolution, self.valid_points_thres, self.sub_sample_factor
            )

            def inv_contract(x):
                mag = torch.linalg.norm(x, ord=pipeline.model.scene_contraction.order, dim=-1)
                mask = mag >= 1
                x_new = x.clone()
                x_new[mask] = (1 / (2 - mag[mask][..., None])) * (x[mask] / mag[mask][..., None])
                return x_new

            if self.save_visibility_grid:
                offset = torch.linspace(-2.0, 2.0, 512)
                x, y, z = torch.meshgrid(offset, offset, offset, indexing="ij")
                offset_cube = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(coarse_mask.device)
                points = offset_cube[coarse_mask.reshape(-1) > 0]
                points = inv_contract(points)
                save_points("mask.ply", points.cpu().numpy())
                torch.save(coarse_mask, "coarse_mask.pt")

            get_surface_sliding_with_contraction(
                sdf=lambda x: (
                    pipeline.model.field.forward_geonetwork(x)[:, 0] - self.marching_cube_threshold
                ).contiguous(),
                resolution=self.resolution,
                bounding_box_min=self.bounding_box_min,
                bounding_box_max=self.bounding_box_max,
                coarse_mask=coarse_mask,
                output_path=self.output_path,
                simplify_mesh=self.simplify_mesh,
                inv_contraction=inv_contract,
            )
            return

        if self.is_occupancy:
            # for unisurf
            get_surface_occupancy(
                # occupancy_fn=lambda x: torch.sigmoid(
                #     10 * pipeline.model.field.forward_geonetwork(x)[:, 0].contiguous()
                # ),
                occupancy_fn=lambda x: torch.sigmoid(
                    10 * pipeline.model.field.get_density_pts(x)[:,0].contiguous()
                ),
                resolution=self.resolution,
                bounding_box_min=self.bounding_box_min,
                bounding_box_max=self.bounding_box_max,
                level=0.7,
                device=pipeline.model.device,
                output_path=self.output_path,
            )
        else:
            assert self.resolution % 512 == 0
            # for sdf we can multi-scale extraction.
            get_surface_sliding(
                sdf=lambda x: pipeline.model.field.forward_geonetwork(x)[:, 0].contiguous(),
                resolution=self.resolution,
                bounding_box_min=self.bounding_box_min,
                bounding_box_max=self.bounding_box_max,
                coarse_mask=pipeline.model.scene_box.coarse_binary_gird,
                output_path=self.output_path,
                simplify_mesh=self.simplify_mesh,
            )

    def meta_main(self) -> None:
        if self.mesh_list is not None:
            with open(self.mesh_list, 'r') as f:
                meshlist = f.read().splitlines()
            for line in meshlist:
                if line == '':
                    continue
                config, outpath = line.split('\t')
                self.load_config = Path(config)
                self.output_path = Path(outpath)
                if 'nerf' in config:
                    self.is_occupancy = True
                else:
                    self.is_occupancy = False
                try:
                    self.main()
                except:
                    pass
        else:
            self.main()


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[ExtractMesh]).meta_main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ExtractMesh)  # noqa
