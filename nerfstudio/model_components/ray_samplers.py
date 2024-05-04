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
Collection of sampling strategies
"""

import math
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import nerfacc
import torch
# from nerfacc import OccupancyGrid
from nerfacc import OccGridEstimator
from torch import nn
from torchtyping import TensorType
import numpy as np
import scipy
from nerfstudio.utils.math_nerf import spherical_coord

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
# from nerfacc.intersection import ray_aabb_intersect
from nerfacc.grid import _ray_aabb_intersect
import pdb

def get_foreground_mask(ray_samples: RaySamples=None, ray_points=None) -> TensorType:
    """_summary_

    Args:
        ray_samples (RaySamples): _description_
    """
    # TODO support multiple foreground type: box and sphere
    if ray_samples is not None:
        inside_sphere_mask = (ray_samples.frustums.get_start_positions().norm(dim=-1, keepdim=True) < 1.0).float()
    elif ray_points is not None:
        inside_sphere_mask = (ray_points.norm(dim=-1, keepdim=True) < 1.0).float()
        return inside_sphere_mask

class Sampler(nn.Module):
    """Generate Samples

    Args:
        num_samples: number of samples to take
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples

    @abstractmethod
    def generate_ray_samples(self) -> RaySamples:
        """Generate Ray Samples"""

    def forward(self, *args, **kwargs) -> RaySamples:
        """Generate ray samples"""
        return self.generate_ray_samples(*args, **kwargs)


class SpacedSampler(Sampler):
    """Sample points according to a function.

    Args:
        num_samples: Number of samples per ray
        spacing_fn: Function that dictates sample spacing (ie `lambda x : x` is uniform).
        spacing_fn_inv: The inverse of spacing_fn.
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        spacing_fn: Callable,
        spacing_fn_inv: Callable,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.train_stratified = train_stratified
        self.single_jitter = single_jitter
        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
    ) -> RaySamples:
        """Generates position samples accoring to spacing function.

        Args:
            ray_bundle: Rays to generate samples for
            num_samples: Number of samples per ray

        Returns:
            Positions and deltas for samples along a ray
        """
        assert ray_bundle is not None
        assert ray_bundle.nears is not None
        assert ray_bundle.fars is not None

        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_rays = ray_bundle.origins.shape[0]

        bins = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.origins.device)[None, ...]  # [1, num_samples+1]

        # TODO More complicated than it needs to be.
        if self.train_stratified and self.training:
            if self.single_jitter:
                t_rand = torch.rand((num_rays, 1), dtype=bins.dtype, device=bins.device)
            else:
                t_rand = torch.rand((num_rays, num_samples + 1), dtype=bins.dtype, device=bins.device)
            bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
            bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
            bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
            bins = bin_lower + (bin_upper - bin_lower) * t_rand

        s_near, s_far = (self.spacing_fn(x) for x in (ray_bundle.nears.clone(), ray_bundle.fars.clone()))
        spacing_to_euclidean_fn = lambda x: self.spacing_fn_inv(x * s_far + (1 - x) * s_near)
        euclidean_bins = spacing_to_euclidean_fn(bins)  # [num_rays, num_samples+1]

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
        )

        return ray_samples


class UniformSampler(SpacedSampler):
    """Sample uniformly along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class LinearDisparitySampler(SpacedSampler):
    """Sample linearly in disparity along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: 1 / x,
            spacing_fn_inv=lambda x: 1 / x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class SqrtSampler(SpacedSampler):
    """Square root sampler along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=torch.sqrt,
            spacing_fn_inv=lambda x: x**2,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class LogSampler(SpacedSampler):
    """Log sampler along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=torch.log,
            spacing_fn_inv=torch.exp,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class UniformLinDispPiecewiseSampler(SpacedSampler):
    """Piecewise sampler along a ray that allocates the first half of the samples uniformly and the second half
    using linearly in disparity spacing.


    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x)),
            spacing_fn_inv=lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x)),
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class PDFSampler(Sampler):
    """Sample based on probability distribution

    Args:
        num_samples: Number of samples per ray
        train_stratified: Randomize location within each bin during training.
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
        include_original: Add original samples to ray.
        histogram_padding: Amount to weights prior to computing PDF.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified: bool = True,
        single_jitter: bool = False,
        include_original: bool = True,
        histogram_padding: float = 0.01,
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.train_stratified = train_stratified
        self.include_original = include_original
        self.histogram_padding = histogram_padding
        self.single_jitter = single_jitter

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        ray_samples: Optional[RaySamples] = None,
        weights: TensorType[..., "num_samples", 1] = None,
        num_samples: Optional[int] = None,
        eps: float = 1e-5,
    ) -> RaySamples:
        """Generates position samples given a distribution.

        Args:
            ray_bundle: Rays to generate samples for
            ray_samples: Existing ray samples
            weights: Weights for each bin
            num_samples: Number of samples per ray
            eps: Small value to prevent numerical issues.

        Returns:
            Positions and deltas for samples along a ray
        """

        if ray_samples is None or ray_bundle is None:
            raise ValueError("ray_samples and ray_bundle must be provided")

        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_bins = num_samples + 1

        weights = weights[..., 0] + self.histogram_padding

        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(eps - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
        cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        if self.train_stratified and self.training:
            # Stratified samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
            if self.single_jitter:
                rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
            else:
                rand = torch.rand((*cdf.shape[:-1], num_samples + 1), device=cdf.device) / num_bins
            u = u + rand
        else:
            # Uniform samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u + 1.0 / (2 * num_bins)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
        u = u.contiguous()

        assert (
            ray_samples.spacing_starts is not None and ray_samples.spacing_ends is not None
        ), "ray_sample spacing_starts and spacing_ends must be provided"
        assert ray_samples.spacing_to_euclidean_fn is not None, "ray_samples.spacing_to_euclidean_fn must be provided"
        existing_bins = torch.cat(
            [
                ray_samples.spacing_starts[..., 0],
                ray_samples.spacing_ends[..., -1:, 0],
            ],
            dim=-1,
        )

        inds = torch.searchsorted(cdf, u, side="right")
        below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
        above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
        cdf_g0 = torch.gather(cdf, -1, below)
        bins_g0 = torch.gather(existing_bins, -1, below)
        cdf_g1 = torch.gather(cdf, -1, above)
        bins_g1 = torch.gather(existing_bins, -1, above)

        t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        bins = bins_g0 + t * (bins_g1 - bins_g0)

        if self.include_original:
            bins, _ = torch.sort(torch.cat([existing_bins, bins], -1), -1)

        # Stop gradients
        bins = bins.detach()

        euclidean_bins = ray_samples.spacing_to_euclidean_fn(bins)

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=ray_samples.spacing_to_euclidean_fn,
        )

        return ray_samples


class VolumetricSampler(Sampler):
    """Sampler inspired by the one proposed in the Instant-NGP paper.
    Generates samples along a ray by sampling the occupancy field.
    Optionally removes occluded samples if the density_fn is provided.
    Args:
    occupancy_grid: Occupancy grid to sample from.
    density_fn: Function that evaluates density at a given point.
    scene_aabb: Axis-aligned bounding box of the scene, should be set to None if the scene is unbounded.
    """

    def __init__(
        self,
        occupancy_grid: Optional[OccGridEstimator] = None,
        density_fn: Optional[Callable[[TensorType[..., 3]], TensorType[..., 1]]] = None,
        scene_aabb: Optional[TensorType[2, 3]] = None,
    ) -> None:

        super().__init__()
        self.scene_aabb = scene_aabb
        self.density_fn = density_fn
        self.occupancy_grid = occupancy_grid
        if self.scene_aabb is not None:
            self.scene_aabb = self.scene_aabb.to("cuda").flatten()
        print(self.scene_aabb)

    def get_sigma_fn(self, origins, directions) -> Optional[Callable]:
        """Returns a function that returns the density of a point.
        Args:
            origins: Origins of rays
            directions: Directions of rays
        Returns:
            Function that returns the density of a point or None if a density function is not provided.
        """

        if self.density_fn is None or not self.training:
            return None

        density_fn = self.density_fn

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = directions[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
            return density_fn(positions)

        return sigma_fn

    def generate_ray_samples(self) -> RaySamples:
        raise RuntimeError(
            "The VolumetricSampler fuses sample generation and density check together. Please call forward() directly."
        )

    # pylint: disable=arguments-differ
    def forward(
        self,
        ray_bundle: RayBundle,
        render_step_size: float,
        near_plane: float = 0.0,
        far_plane: Optional[float] = None,
        cone_angle: float = 0.0,
        alpha_thre: float = 1e-2,
    ) -> Tuple[RaySamples, TensorType["total_samples",]]:
        """Generate ray samples in a bounding box.
        Args:
            ray_bundle: Rays to generate samples for
            render_step_size: Minimum step size to use for rendering
            near_plane: Near plane for raymarching
            far_plane: Far plane for raymarching
            cone_angle: Cone angle for raymarching, set to 0 for uniform marching.
            alpha_thre: Threshold for ray marching
        Returns:
            a tuple of (ray_samples, packed_info, ray_indices)
            The ray_samples are packed, only storing the valid samples.
            The ray_indices contains the indices of the rays that each sample belongs to.
        """

        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()
        if ray_bundle.camera_indices is not None:
            camera_indices = ray_bundle.camera_indices.contiguous()
        else:
            camera_indices = None

        ray_indices, starts, ends = self.occupancy_grid.sampling(
            rays_o=rays_o,
            rays_d=rays_d,
            # scene_aabb=self.scene_aabb,
            # grid=self.occupancy_grid,
            # this is a workaround - using density causes crash and damage quality. should be fixed
            sigma_fn=None,  # self.get_sigma_fn(rays_o, rays_d),
            render_step_size=render_step_size,
            near_plane=near_plane,
            far_plane=far_plane,
            stratified=self.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        num_samples = starts.shape[0]
        if num_samples == 0:
            # create a single fake sample and update packed_info accordingly
            # this says the last ray in packed_info has 1 sample, which starts and ends at 1
            ray_indices = torch.zeros((1,), dtype=torch.long, device=rays_o.device)
            starts = torch.ones((1, 1), dtype=starts.dtype, device=rays_o.device)
            ends = torch.ones((1, 1), dtype=ends.dtype, device=rays_o.device)

        origins = rays_o[ray_indices]
        dirs = rays_d[ray_indices]
        if camera_indices is not None:
            camera_indices = camera_indices[ray_indices]

        zeros = torch.zeros_like(origins[:, :1])
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=dirs,
                starts=starts,
                ends=ends,
                pixel_area=zeros,
            ),
            camera_indices=camera_indices,
        )
        return ray_samples, ray_indices


class ProposalNetworkSampler(Sampler):
    """Sampler that uses a proposal network to generate samples."""

    def __init__(
        self,
        num_proposal_samples_per_ray: Tuple[int] = (64,),
        num_nerf_samples_per_ray: int = 32,
        num_proposal_network_iterations: int = 2,
        use_uniform_sampler: bool = False,
        single_jitter: bool = False,
        update_sched: Callable = lambda x: 1,
    ) -> None:
        super().__init__()
        self.num_proposal_samples_per_ray = num_proposal_samples_per_ray
        self.num_nerf_samples_per_ray = num_nerf_samples_per_ray
        self.num_proposal_network_iterations = num_proposal_network_iterations
        self.update_sched = update_sched
        if self.num_proposal_network_iterations < 1:
            raise ValueError("num_proposal_network_iterations must be >= 1")

        if use_uniform_sampler:
            self.initial_sampler = UniformSampler(single_jitter=single_jitter)
        else:
            self.initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)

        self.pdf_sampler = PDFSampler(include_original=False, single_jitter=single_jitter)

        self._anneal = 1.0
        self._steps_since_update = 0
        self._step = 0

    def set_anneal(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._anneal = anneal

    def step_cb(self, step):
        """Callback to register a training step has passed. This is used to keep track of the sampling schedule"""
        self._step = step
        self._steps_since_update += 1

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        density_fns: Optional[List[Callable]] = None,
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        assert density_fns is not None

        weights_list = []
        ray_samples_list = []

        n = self.num_proposal_network_iterations
        weights = None
        ray_samples = None
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)
            if is_prop:
                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    density = density_fns[i_level](ray_samples.frustums.get_positions())
                else:
                    with torch.no_grad():
                        density = density_fns[i_level](ray_samples.frustums.get_positions())
                weights = ray_samples.get_weights(density)
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)
        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list


class ErrorBoundedSampler(Sampler):
    """VolSDF's error bounded sampler that uses a sdf network to generate samples."""

    def __init__(
        self,
        num_samples: int = 64,
        num_samples_eval: int = 128,
        num_samples_extra: int = 32,
        eps: float = 0.1,
        beta_iters: int = 10,
        max_total_iters: int = 5,
        add_tiny: float = 1e-6,
        single_jitter: bool = False,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.num_samples_eval = num_samples_eval
        self.num_samples_extra = num_samples_extra
        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.add_tiny = add_tiny
        self.single_jitter = single_jitter

        # samplers
        self.uniform_sampler = UniformSampler(single_jitter=single_jitter)
        self.pdf_sampler = PDFSampler(
            include_original=False,
            single_jitter=single_jitter,
            histogram_padding=1e-5,
        )

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        density_fn: Optional[Callable] = None,
        sdf_fn: Optional[Callable] = None,
        return_eikonal_points: bool = True,
    ) -> Union[Tuple[RaySamples, torch.Tensor], RaySamples]:
        assert ray_bundle is not None
        assert density_fn is not None
        assert sdf_fn is not None

        beta0 = density_fn.get_beta().detach()

        # Start with uniform sampling
        ray_samples = self.uniform_sampler(ray_bundle, num_samples=self.num_samples_eval)

        # Get maximum beta from the upper bound (Lemma 2)
        deltas = ray_samples.deltas.squeeze(-1)

        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (deltas**2.0).sum(-1)
        beta = torch.sqrt(bound)

        total_iters, not_converge = 0, True
        sorted_index = None
        new_samples = ray_samples

        # Algorithm 1
        while not_converge and total_iters < self.max_total_iters:

            with torch.no_grad():
                new_sdf = sdf_fn(new_samples)

            # merge sdf predictions
            if sorted_index is not None:
                sdf_merge = torch.cat([sdf.squeeze(-1), new_sdf.squeeze(-1)], -1)
                sdf = torch.gather(sdf_merge, 1, sorted_index).unsqueeze(-1)
            else:
                sdf = new_sdf

            # Calculating the bound d* (Theorem 1)
            d_star = self.get_dstar(sdf, ray_samples)

            # Updating beta using line search
            beta = self.get_updated_beta(beta0, beta, density_fn, sdf, d_star, ray_samples)

            # Upsample more points
            density = density_fn(sdf.reshape(ray_samples.shape), beta=beta.unsqueeze(-1))

            weights, transmittance = ray_samples.get_weights_and_transmittance(density.unsqueeze(-1))

            #  Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0

            if not_converge and total_iters < self.max_total_iters:
                # Sample more points proportional to the current error bound
                deltas = ray_samples.deltas.squeeze(-1)

                error_per_section = (
                    torch.exp(-d_star / beta.unsqueeze(-1)) * (deltas**2.0) / (4 * beta.unsqueeze(-1) ** 2)
                )

                error_integral = torch.cumsum(error_per_section, dim=-1)
                weights = (torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0) * transmittance[..., 0]

                new_samples = self.pdf_sampler(
                    ray_bundle, ray_samples, weights.unsqueeze(-1), num_samples=self.num_samples_eval
                )

                ray_samples, sorted_index = self.merge_ray_samples(ray_bundle, ray_samples, new_samples)

            else:
                # Sample the final sample set to be used in the volume rendering integral
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, weights, num_samples=self.num_samples)

        if return_eikonal_points:
            # sample some of the near surface points for eikonal loss
            sampled_points = ray_samples.frustums.get_positions().view(-1, 3)
            idx = torch.randint(sampled_points.shape[0], (ray_samples.shape[0] * 10,)).to(sampled_points.device)
            points = sampled_points[idx]

        # Add extra samples uniformly
        if self.num_samples_extra > 0:
            ray_samples_uniform = self.uniform_sampler(ray_bundle, num_samples=self.num_samples_extra)
            ray_samples, _ = self.merge_ray_samples(ray_bundle, ray_samples, ray_samples_uniform)

        if return_eikonal_points:
            return ray_samples, points

        return ray_samples

    def get_dstar(self, sdf, ray_samples: RaySamples):
        """Calculating the bound d* (Theorem 1) from VolSDF"""
        d = sdf.reshape(ray_samples.shape)
        dists = ray_samples.deltas.squeeze(-1)
        a, b, c = dists[:, :-1], d[:, :-1].abs(), d[:, 1:].abs()
        first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
        second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
        d_star = torch.zeros(ray_samples.shape[0], ray_samples.shape[1] - 1).to(d.device)
        d_star[first_cond] = b[first_cond]
        d_star[second_cond] = c[second_cond]
        s = (a + b + c) / 2.0
        area_before_sqrt = s * (s - a) * (s - b) * (s - c)
        mask = ~first_cond & ~second_cond & (b + c - a > 0)
        d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
        d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

        # padding to make the same shape as ray_samples
        # d_star_left = torch.cat((d_star[:, :1], d_star), dim=-1)
        # d_star_right = torch.cat((d_star, d_star[:, -1:]), dim=-1)
        # d_star = torch.minimum(d_star_left, d_star_right)

        d_star = torch.cat((d_star, d_star[:, -1:]), dim=-1)
        return d_star

    def get_updated_beta(self, beta0, beta, density_fn, sdf, d_star, ray_samples: RaySamples):
        curr_error = self.get_error_bound(beta0, density_fn, sdf, d_star, ray_samples)
        beta[curr_error <= self.eps] = beta0
        beta_min, beta_max = beta0.repeat(ray_samples.shape[0]), beta
        for j in range(self.beta_iters):
            beta_mid = (beta_min + beta_max) / 2.0
            curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), density_fn, sdf, d_star, ray_samples)
            beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
            beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
        beta = beta_max
        return beta

    def get_error_bound(self, beta, density_fn, sdf, d_star, ray_samples):
        """Get error bound from VolSDF"""
        densities = density_fn(sdf.reshape(ray_samples.shape), beta=beta)

        deltas = ray_samples.deltas.squeeze(-1)
        delta_density = deltas * densities

        integral_estimation = torch.cumsum(delta_density[..., :-1], dim=-1)
        integral_estimation = torch.cat(
            [torch.zeros((*integral_estimation.shape[:1], 1), device=densities.device), integral_estimation], dim=-1
        )

        error_per_section = torch.exp(-d_star / beta) * (deltas**2.0) / (4 * beta**2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0) * torch.exp(-integral_estimation)

        return bound_opacity.max(-1)[0]

    def merge_ray_samples(self, ray_bundle: RayBundle, ray_samples_1: RaySamples, ray_samples_2: RaySamples):
        """Merge two set of ray samples and return sorted index which can be used to merge sdf values

        Args:
            ray_samples_1 : ray_samples to merge
            ray_samples_2 : ray_samples to merge
        """

        starts_1 = ray_samples_1.spacing_starts[..., 0]
        starts_2 = ray_samples_2.spacing_starts[..., 0]

        ends = torch.maximum(ray_samples_1.spacing_ends[..., -1:, 0], ray_samples_2.spacing_ends[..., -1:, 0])

        bins, sorted_index = torch.sort(torch.cat([starts_1, starts_2], -1), -1)

        bins = torch.cat([bins, ends], dim=-1)

        # Stop gradients
        bins = bins.detach()

        euclidean_bins = ray_samples_1.spacing_to_euclidean_fn(bins)

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=ray_samples_1.spacing_to_euclidean_fn,
        )

        return ray_samples, sorted_index


def save_points(path_save, pts, colors=None, normals=None, BRG2RGB=False):
    """save points to point cloud using open3d"""
    assert len(pts) > 0
    if colors is not None:
        assert colors.shape[1] == 3
    assert pts.shape[1] == 3
    import numpy as np
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        # Open3D assumes the color values are of float type and in range [0, 1]
        if np.max(colors) > 1:
            colors = colors / np.max(colors)
        if BRG2RGB:
            colors = np.stack([colors[:, 2], colors[:, 1], colors[:, 0]], axis=-1)
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)

    o3d.io.write_point_cloud(path_save, cloud)

class NeRFAccSampler(Sampler):
    """Sampler inspired by the one proposed in the Instant-NGP paper.
    Generates samples along a ray by sampling the occupancy field.
    Optionally removes occluded samples if the density_fn is provided.
    Args:
    occupancy_grid: Occupancy grid to sample from.
    density_fn: Function that evaluates density at a given point.
    scene_aabb: Axis-aligned bounding box of the scene, should be set to None if the scene is unbounded.
    """

    def __init__(
        self,
        occupancy_grid: Optional[OccGridEstimator] = None,
        density_fn: Optional[Callable[[TensorType[..., 3]], TensorType[..., 1]]] = None,
        assist_grid=None,
        uniform_sampler = None,
        scene_aabb: Optional[TensorType[2, 3]] = None,
        steps_warmup: int = 256, #1200,
        steps_per_grid_update: int = 16, #600,
        topk_iter: int = 4000,
        render_step_size = 0.01,
        surface_threshold: float=25.0, 
        pdb_debug=False,
    ) -> None:
        super().__init__()
        self.steps_warmup = steps_warmup
        self.steps_per_grid_update = steps_per_grid_update

        self.decay = 0.95
        self.topk_iter = topk_iter
        self.surface_threshold = surface_threshold

        self.scene_aabb = scene_aabb
        self.density_fn = density_fn
        self.occupancy_grid = occupancy_grid
        self.render_step_size = render_step_size
        self.pdb_debug=pdb_debug

        self.uniform_sampler = uniform_sampler
        self.assist_grid = assist_grid
        self.register_buffer("_alpha", torch.ones(self.assist_grid.binaries.shape, dtype=torch.float)*0.5)
        self.register_buffer("_update_counter", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("_uncertainty_binary", torch.ones(self.occupancy_grid.binaries.shape, dtype=torch.float))

    @torch.no_grad()
    def cal_multiCam(self, train_set, candidates, topk_list, candidate_cam_info, dist_threshold=1.732, method='dist', cand_topk=4):
        # sampling from occupancy grids
        """
        return_dict['poses'] = poses
        return_dict['img_w_h'] = (img_w, img_h)
        return_dict['fx_fy_cx_cy'] = K
        """
        candidates = np.array(candidates)
        if method=='topk':  
            topk_add = torch.topk(topk_list, cand_topk)[1].cpu().numpy()
            topk_add = candidates[topk_add.astype(int)]
            return topk_add
        train_set = np.array(train_set)
        poses = candidate_cam_info['poses'].detach().cpu().numpy() # [100, 3, 4]
        return_ids = -np.ones(cand_topk, dtype=np.int32)

        sorted, indices =torch.sort(topk_list, dim=0, descending=True, stable=True)
        indices = indices.detach().cpu().numpy() #(96, 1) -> (96,)
        
        next_idx= candidates[indices[0]]
        return_ids[0] = next_idx
        indices = indices[1:]

        for i in range(cand_topk-1):
            train_set = np.append(train_set, next_idx)
            # candidates = np.delete(candidates, next_idx)
            train_poses_xyz = np.squeeze(poses[train_set][..., 3:], -1)
            # divide sphere by topk
            pose_indices = candidates[indices]
            cand_poses_xyz = np.squeeze(poses[pose_indices][..., 3:], -1)  #(96, 3, 1) (13, 2, 3) when cascade is > 1.

            dist = scipy.spatial.distance.cdist(train_poses_xyz, cand_poses_xyz) # (5, 95)
            print("dist threshold mean", np.mean(dist))
            cand_thres = np.all(dist>=dist_threshold, axis=0, keepdims=False)
            while np.count_nonzero(cand_thres) ==0:
                dist_threshold = self.decay*dist_threshold
                print("REDUCING DIST THRESHOLD !!!!", dist_threshold)
                cand_thres = np.all(dist>=dist_threshold, axis=0, keepdims=False)
            next_idx = pose_indices[cand_thres][0]
            return_ids[i+1] = next_idx
            indices = np.delete(indices, np.where(indices == indices[cand_thres][0]))
        return return_ids 
    
    def update_step(self, step):
        self.step=step

    @torch.no_grad()
    def cal_entropy(self, inputs, ray_indices, occupancy_points, entropy_fn, 
                    type='ent', surface_points=None, assist_grid=None, scene_aabb=None, mean=True):
        if type== 'no_surface':
            surface_grids = occupancy_points
        else:
            # TODO fix the mask threshold 0.36->0.8            
            surface_mask = \
                torch.where(assist_grid.occs.reshape(assist_grid.binaries.shape)>0.8, 1.0, 0.0)
            # surface_points = nerfacc.grid.query_grid(inputs, self.grid.roi_aabb, 
            #                                         surface_mask, self.grid.contraction_type)
            if surface_points is None:
                surface_points, surface_pnt_mask = nerfacc.grid._query(inputs, surface_mask, 
                                                                   scene_aabb)
            ray_wsurface =set(ray_indices[surface_points>0].detach().cpu().numpy())
            ray_wosurface = set(ray_indices[surface_points==0].detach().cpu().numpy()) - ray_wsurface

            surface_grids = occupancy_points*surface_points
        
        acquisition_value = entropy_fn(surface_grids)
        acq_full = torch.where(surface_grids>0, acquisition_value, 0.0)
        if type=='ent' and len(ray_wosurface)>0:
            for ele in ray_wosurface:
                extra_acquisition_value = entropy_fn(occupancy_points)
                acq_extra = torch.where(ray_indices==ele, extra_acquisition_value, 0.0)
                acq_full = torch.sum(torch.stack([acq_full, acq_extra], 0), 0)
        if mean:
            acq_full = torch.mean(acq_full)
        return acq_full, surface_points 

    @torch.no_grad()
    def eval_k_views(self, ray_bundle, render_step_size, 
                     near_plane, far_plane,
                    cone_angle, alpha_sample_thre, type='ent', grid_sampling=True):
        # sampling from occupancy grids
        # ray_bundle shape (1024,)
        def make_entropy(val):
            # you can delete 1e-6 when the self.alpha_sample_thre is low enough.
            val = val.clip(1e-6, 1.0- (1e-6))
            return -val*torch.log(val) -(1-val)*torch.log(1-val)
        # TODO how about neus_sampler?
        
        if grid_sampling:
            ray_samples, ray_indices = self(ray_bundle, render_step_size, near_plane, far_plane,
                                            cone_angle, alpha_sample_thre)
        else:
            ray_samples = self.uniform_sampler(ray_bundle)
            n_rays, n_samples = ray_samples.shape
            ray_indices = torch.arange(n_rays)
            ray_indices = ray_indices.repeat_interleave(n_samples)
        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        # occupancy_points = nerfacc.grid.query_grid(inputs, self.grid.roi_aabb, 
        #                                         self._alpha, self.grid.contraction_type)
        occupancy_points, occup_mask = nerfacc.grid._query(inputs, self._alpha, self.scene_aabb)
        occupancy_acq, surface_points = self.cal_entropy(inputs, ray_indices, occupancy_points,
                                                          make_entropy, type, None,
                                                          self.assist_grid, self.scene_aabb)
        # acquisition_value = make_entropy(occupancy_points)
        # # acquisition_value = torch.mean(acquisition_value)
        # acq_full = torch.mean(acquisition_value)
        # return acq_full, torch.mean(occupancy_points)
        return occupancy_acq, torch.mean(occupancy_points)
    
    @torch.no_grad()
    def initialize_grid(self, camera_info, chunk, near_plane):
        camera_to_worlds = camera_info['c2w']
        width = camera_info['width']
        height = camera_info['height']
        K = camera_info['K']
        self.assist_grid.mark_invisible_cells(K, camera_to_worlds, width, height, near_plane, chunk)
        # self.bg_grid.mark_invisible_cells(K, camera_to_worlds, width, height, near_plane, chunk)

    def get_sigma_fn(self, origins, directions) -> Optional[Callable]:
        """Returns a function that returns the density of a point.
        Args:
            origins: Origins of rays
            directions: Directions of rays
        Returns:
            Function that returns the density of a point or None if a density function is not provided.
        """

        if self.density_fn is None or not self.training:
            return None

        density_fn = self.density_fn

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = directions[ray_indices]
            # positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            return torch.squeeze(density_fn(positions, steps=self.step))

        return sigma_fn
    
    def sigma_position(self, positions):
        return torch.squeeze(self.density_fn(positions, steps=self.step))
    
    def surface_pts(self, points):
        density= torch.squeeze(self.density_fn(points, steps=self.step))
        # import pdb
        # pdb.set_trace()
        threshold = torch.min(torch.mean(density), 
                              torch.Tensor([self.surface_threshold]).to(density.device))
        mask = (density > threshold).float()
        return mask

    @torch.no_grad()
    def update_binary_grid(self, step):
        self.occupancy_grid.update_every_n_steps(step=step, 
                                occ_eval_fn=self.sigma_position, 
                                occ_thre=0.01,
                                ema_decay = self.decay,
                                warmup_steps = self.steps_warmup,
                                n = self.steps_per_grid_update,
                                )
        self.assist_grid.update_every_n_steps(step=step, 
                                occ_eval_fn=self.surface_pts, 
                                occ_thre=0.0,
                                ema_decay = self.decay,
                                warmup_steps = self.steps_warmup,
                                n = self.steps_per_grid_update,
                                )

        if step % self.steps_per_grid_update == 0:
            flag = (self.assist_grid.occs<0).reshape(self.assist_grid.binaries.shape)
            delta_density = self.render_step_size * self.occupancy_grid.occs
            alphas = 1 - torch.exp(-delta_density)
            self._alpha = \
                torch.where(flag,
                        self._alpha,
                        alphas.reshape(self.assist_grid.binaries.shape))
             

        if step > self.steps_warmup and step % self.steps_per_grid_update == 0:
            self._update_counter += 1
        if step<(self.topk_iter*5-3) and step%self.topk_iter ==0:
            self._update_counter = torch.zeros(1, dtype=torch.int32)
            step = step%self.topk_iter
    

    # pylint: disable=arguments-differ
    def forward(
        self,
        ray_bundle: RayBundle,
        render_step_size: float,
        near_plane: float = 0.0,
        far_plane: Optional[float] = None,
        cone_angle: float = 0.0,
        alpha_sample_thre: float = 1e-2,
    ) -> Tuple[RaySamples, TensorType["total_samples",]]:
        """Generate ray samples in a bounding box.
        Args:
            ray_bundle: Rays to generate samples for
            render_step_size: Minimum step size to use for rendering
            near_plane: Near plane for raymarching
            far_plane: Far plane for raymarching
            cone_angle: Cone angle for raymarching, set to 0 for uniform marching.
            alpha_sample_thre: Threshold for ray marching
        Returns:
            a tuple of (ray_samples, packed_info, ray_indices)
            The ray_samples are packed, only storing the valid samples.
            The ray_indices contains the indices of the rays that each sample belongs to.
        """
        if self._update_counter.item() <= 0:
            return self.uniform_sampler(ray_bundle)

        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()
        times = ray_bundle.times

        if ray_bundle.nears is not None and ray_bundle.fars is not None:
            t_min = ray_bundle.nears.contiguous().reshape(-1)
            t_max = ray_bundle.fars.contiguous().reshape(-1)
        else:
            t_min = None
            t_max = None

        if far_plane is None:
            far_plane = 1e10

        if ray_bundle.camera_indices is not None:
            camera_indices = ray_bundle.camera_indices.contiguous()
        else:
            camera_indices = None

        ray_indices, starts, ends = self.occupancy_grid.sampling(
            rays_o=rays_o,
            rays_d=rays_d,
            t_min=t_min,
            t_max=t_max,
            sigma_fn=self.get_sigma_fn(rays_o, rays_d),
            render_step_size=render_step_size,
            near_plane=near_plane,
            far_plane=far_plane,
            stratified=self.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_sample_thre,
        )
        num_samples = starts.shape[0]
        if num_samples == 0:
            # create a single fake sample and update packed_info accordingly
            # this says the last ray in packed_info has 1 sample, which starts and ends at 1
            ray_indices = torch.zeros((1,), dtype=torch.long, device=rays_o.device)
            starts = torch.ones((1,), dtype=starts.dtype, device=rays_o.device)
            ends = torch.ones((1,), dtype=ends.dtype, device=rays_o.device)

        origins = rays_o[ray_indices]
        dirs = rays_d[ray_indices]
        if camera_indices is not None:
            camera_indices = camera_indices[ray_indices]

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=dirs,
                starts=starts[..., None],
                ends=ends[..., None],
                pixel_area=ray_bundle[ray_indices].pixel_area,
            ),
            camera_indices=camera_indices,
        )
        if ray_bundle.times is not None:
            ray_samples.times = ray_bundle.times[ray_indices]
        return ray_samples, ray_indices


class NeuSSampler(Sampler):
    """NeuS sampler that uses a sdf network to generate samples with fixed variance value in each iterations."""

    def __init__(
        self,
        num_samples: int = 64,
        num_samples_importance: int = 64,
        num_samples_outside: int = 32,
        num_upsample_steps: int = 4,
        base_variance: float = 64,
        single_jitter: bool = True,
        pdb_debug=False,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.num_samples_importance = num_samples_importance
        self.num_samples_outside = num_samples_outside
        self.num_upsample_steps = num_upsample_steps
        self.base_variance = base_variance
        self.single_jitter = single_jitter
        self.pdb_debug = pdb_debug

        # samplers
        self.uniform_sampler = UniformSampler(single_jitter=single_jitter)
        self.pdf_sampler = PDFSampler(
            include_original=False,
            single_jitter=single_jitter,
            histogram_padding=1e-5,
        )
        self.outside_sampler = UniformSampler(single_jitter=single_jitter)
        # self.outside_sampler = LinearDisparitySampler()
        # TODO make it outside
        # for merge samples
        self.error_bounded_sampler = ErrorBoundedSampler()

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        sdf_fn: Optional[Callable] = None,
        ray_samples: Optional[RaySamples] = None,
        steps=None,
    ) -> Union[Tuple[RaySamples, torch.Tensor], RaySamples]:
        assert ray_bundle is not None
        assert sdf_fn is not None

        # Start with uniform sampling
        if ray_samples is None:
            ray_samples = self.uniform_sampler(ray_bundle, num_samples=self.num_samples)

        total_iters = 0
        sorted_index = None
        new_samples = ray_samples

        base_variance = self.base_variance
        if self.pdb_debug:
            pdb.set_trace()

        while total_iters < self.num_upsample_steps:

            with torch.no_grad():
                new_sdf = sdf_fn(new_samples, steps=steps)

            # merge sdf predictions
            if sorted_index is not None:
                sdf_merge = torch.cat([sdf.squeeze(-1), new_sdf.squeeze(-1)], -1)
                sdf = torch.gather(sdf_merge, 1, sorted_index).unsqueeze(-1)
            else:
                sdf = new_sdf

            # compute with fix variances
            alphas = self.rendering_sdf_with_fixed_inv_s(
                ray_samples, sdf.reshape(ray_samples.shape), inv_s=base_variance * 2**total_iters
            )

            weights = ray_samples.get_weights_from_alphas(alphas[..., None])
            weights = torch.cat((weights, torch.zeros_like(weights[:, :1])), dim=1)
            if self.pdb_debug:
                # TODO Check whether weights at the end is always zero 
                pdb.set_trace()
            new_samples = self.pdf_sampler(
                ray_bundle,
                ray_samples,
                weights,
                num_samples=self.num_samples_importance // self.num_upsample_steps,
            )
            if self.pdb_debug:
                # TODO new_samples avg deltas 57, uniform_sample avg deltas 15
                pdb.set_trace()

            ray_samples, sorted_index = self.error_bounded_sampler.merge_ray_samples(
                ray_bundle, ray_samples, new_samples
            )
            if self.pdb_debug:
                # TODO Check whether sorted_index is really sorted.
                # TODO during merge, deltas are corrected to be 15, and 
                # sorted_index only includes previous ray_samples index
                pdb.set_trace()

            total_iters += 1

        # save_points("p.ply", ray_samples.frustums.get_start_positions().detach().cpu().numpy().reshape(-1, 3))
        # exit(-1)
        # TODO
        # sample more points outside surface
        if self.num_samples_outside > 0:
            ray_samples_uniform_outside = self.outside_sampler(ray_bundle, num_samples=self.num_samples_outside)
            # merge
            ray_samples, _ = self.error_bounded_sampler.merge_ray_samples(
                ray_bundle, ray_samples, ray_samples_uniform_outside
            )
        if self.pdb_debug:
            pdb.set_trace()

        return ray_samples

    def rendering_sdf_with_fixed_inv_s(self, ray_samples: RaySamples, sdf: torch.Tensor, inv_s):
        """rendering given a fixed inv_s as NeuS"""
        batch_size = ray_samples.shape[0]
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        deltas = ray_samples.deltas[:, :-1, 0]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (deltas + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=sdf.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0)

        dist = deltas
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

        return alpha

class NeuSAccSampler(Sampler):
    """Voxel surface guided sampler in NeuralReconW."""

    def __init__(
        self,
        aabb,
        grid=None,
        bg_grid=None,
        assist_grid=None,
        bg_assist_grid=None,
        scene_aabb=None,
        bg_aabb=None,

        neus_sampler: NeuSSampler = None,
        resolution: int = 128,
        bg_resolution=64,

        steps_warmup: int = 256, #1200,
        steps_per_grid_update: int = 16, #600,
        coarse_sample: int = 64,
        topk_iter: int = 4000,

        density_fn=None,
        sdf_fn=None,
        inv_s_fn=None,
        pdb_debug = False,
    ) -> None:
        super().__init__()
        self.aabb = aabb
        self.steps_warmup = steps_warmup
        self.steps_per_grid_update = steps_per_grid_update
        # To be updated by step update function
        self.step_size = 1.732 * 2 * aabb[1, 0] / coarse_sample
        self.decay = 0.95
        self.topk_iter = topk_iter

        self.density_fn = density_fn
        self.sdf_fn = sdf_fn
        self.inv_s_fn = inv_s_fn
        self.pdb_debug = pdb_debug

        # only supports cubic bbox for now
        assert aabb[0, 0] == aabb[0, 1] and aabb[0, 0] == aabb[0, 2]
        assert aabb[1, 0] == aabb[1, 1] and aabb[1, 0] == aabb[1, 2]
        self.grid_size = resolution
        self.bg_grid_size = bg_resolution
        
        # self.voxel_size = (aabb[1, 0] - aabb[0, 0]) / self.grid_size

        # nesu_sampler at the begining of training
        self.neus_sampler = neus_sampler
        self.uniform_sampler = UniformSampler(coarse_sample)

        self.grid = grid
        self.bg_grid = bg_grid
        self.assist_grid=assist_grid
        self.bg_assist_grid = bg_assist_grid

        self.scene_aabb = scene_aabb
        self.bg_aabb =bg_aabb
        # import pdb
        # pdb.set_trace()
        self.bg_flag = False
        # self.grid = nerfacc.OccupancyGrid(aabb.reshape(-1), resolution=self.resolution)
        if density_fn is not None:
            self.bg_flag = True
            self.register_buffer("bg_alpha", torch.ones(self.bg_grid.binaries.shape, dtype=torch.float)*0.5)

        # self.register_buffer("_alpha", torch.ones(self.grid.binaries.squeeze().shape, dtype=torch.float)*0.5)
        self.register_buffer("_alpha", torch.ones(self.grid.binaries.shape, dtype=torch.float)*0.5)
        self.register_buffer("_update_counter", torch.zeros(1, dtype=torch.int32))

        if self.bg_flag:
            cubes = []
            for j in range(len(self.bg_grid.aabbs)):
                aabb = self.bg_grid.aabbs[j].reshape(-1, 3)
                voxel_size = (aabb[1, 0] - aabb[0, 0]) / self.bg_grid_size
                cube_coordinate = self.init_grid_coordinate(aabb, voxel_size, self.bg_grid_size)
                cubes.append(cube_coordinate)
            self.register_buffer("bg_cube_coordinate", torch.stack(cubes, dim=0))

        voxel_size = (aabb[1, 0] - aabb[0, 0]) / self.grid_size
        cube_coordinate = self.init_grid_coordinate(aabb, voxel_size, self.grid_size)
        self.register_buffer("cube_coordinate", cube_coordinate)

    def init_grid_coordinate(self, aabb, voxel_size, grid_size):
        # coarse grid coordinates
        offset_x = torch.linspace(
            aabb[0, 0] + voxel_size / 2.0, aabb[1, 0] - voxel_size / 2.0, grid_size
        )
        offset_y = torch.linspace(
            aabb[0, 1] + voxel_size / 2.0, aabb[1, 1] - voxel_size / 2.0, grid_size
        )
        offset_z = torch.linspace(
            aabb[0, 2] + voxel_size / 2.0, aabb[1, 2] - voxel_size / 2.0, grid_size
        )
        x, y, z = torch.meshgrid(offset_x, offset_y, offset_z, indexing="ij")
        cube_coordinate = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        return cube_coordinate

    def update_step_size(self, step, inv_s=None):
        assert inv_s is not None
        inv_s = inv_s().item()
        self.step_size = 14.0 / inv_s / 16
        self.step = step

    @torch.no_grad()
    def cal_multiCam(self, train_set, candidates, topk_list, candidate_cam_info, 
                     dist_threshold=1.732, method='dist', cand_topk=4):
        # sampling from occupancy grids
        """
        return_dict['poses'] = poses
        return_dict['img_w_h'] = (img_w, img_h)
        return_dict['fx_fy_cx_cy'] = K
        """
        candidates = np.array(candidates)
        if method=='topk':  
            topk_add = torch.topk(topk_list, cand_topk)[1].cpu().numpy()
            topk_add = candidates[topk_add.astype(int)]
            return topk_add
        train_set = np.array(train_set)
        poses = candidate_cam_info['poses'].detach().cpu().numpy() # [100, 3, 4]
        return_ids = -np.ones(cand_topk, dtype=np.int32)

        sorted, indices =torch.sort(topk_list, dim=0, descending=True, stable=True)
        indices = indices.detach().cpu().numpy() #(96, 1) -> (96,)
        
        next_idx= candidates[indices[0]]
        return_ids[0] = next_idx
        indices = indices[1:]

        for i in range(cand_topk-1):
            train_set = np.append(train_set, next_idx)
            # candidates = np.delete(candidates, next_idx)
            train_poses_xyz = np.squeeze(poses[train_set][..., 3:], -1)
            # divide sphere by topk
            pose_indices = candidates[indices]
            cand_poses_xyz = np.squeeze(poses[pose_indices][..., 3:], -1)  #(96, 3, 1) (13, 2, 3) when cascade is > 1.

            dist = scipy.spatial.distance.cdist(train_poses_xyz, cand_poses_xyz) # (5, 95)
            print("dist threshold mean", np.mean(dist))
            cand_thres = np.all(dist>=dist_threshold, axis=0, keepdims=False)
            while np.count_nonzero(cand_thres) ==0:
                dist_threshold = self.decay*dist_threshold
                print("REDUCING DIST THRESHOLD !!!!", dist_threshold)
                cand_thres = np.all(dist>=dist_threshold, axis=0, keepdims=False)
            next_idx = pose_indices[cand_thres][0]
            return_ids[i+1] = next_idx
            indices = np.delete(indices, np.where(indices == indices[cand_thres][0]))
        return return_ids

    @torch.no_grad()
    def cal_entropy_experiment(self, inputs, ray_indices, occupancy_points, entropy_fn, 
                    type='ent', surface_points=None, assist_grid=None, scene_aabb=None, starts=None):
        # starts = starts/starts.max()
        # Divide by distance from origin
        occupancy_points = entropy_fn(occupancy_points)/starts[:,0]
        if type== 'no_surface':
            # Since there is no 'surface', the grids has equal contributions 1 divided by the number of samples on the ray.
            surface_points = torch.ones(inputs.shape[0], 1)
            if self.packed_info is not None:
                surface_points /= torch.repeat_interleave(self.packed_info[:,1],self.packed_info[:,1] )
        else:
            surf_dict = {}
            surface_mask = assist_grid.occs.reshape(assist_grid.binaries.shape)
            if surface_points is None:
                # extract surface from sampled grids
                surface_points, surface_pnt_mask = nerfacc.grid._query(inputs, surface_mask, 
                                                                   scene_aabb)
                # assist grids can have -1 grid, which is not covered by current cameras
                surface_points = surface_points.clip(0.0, 1.0)
                # we call grids as 'surface' when it has surface probability bigger than 0.5 
                ray_wsurface =ray_indices[surface_points>=0.5].detach().cpu().numpy()
                set_ray_wsurface = set(ray_wsurface)
                # infer rays that do not have any surfaces
                ray_wosurface = set(ray_indices[surface_points<0.5].detach().cpu().numpy()) - set_ray_wsurface
                # for rays that do not have any surfaces, all grids on the ray contribute equally
                if len(ray_wosurface)>0:
                    for ele in ray_wosurface:
                        surface_points[ray_indices==ele] = 1/ self.packed_info[ele, 1]
                # count surface grids on rays with surfaces 
                for i in range(len(ray_wsurface)):
                    if ray_wsurface[i] in surf_dict:
                        surf_dict[ray_wsurface[i]] +=1
                    else:
                        surf_dict[ray_wsurface[i]] = 1
                # divide by the number of 'surface' 
                set_ray_wsurface = list(set_ray_wsurface)
                for j in range(len(set_ray_wsurface)):
                    surface_points[self.packed_info[set_ray_wsurface[j],0]:
                                   self.packed_info[set_ray_wsurface[j],0]+self.packed_info[set_ray_wsurface[j],1]] /= surf_dict[set_ray_wsurface[j]]
            # surface_points /= torch.repeat_interleave(self.packed_info[:,1:2],self.packed_info[:,1:2] )
        acq_full = nerfacc.accumulate_along_rays(surface_points, ray_indices=ray_indices, 
                                          values=torch.unsqueeze(occupancy_points, -1), n_rays=self.n_rays)   
        
        acq_full = torch.mean(acq_full)
        return acq_full, surface_points     

    @torch.no_grad()
    def cal_entropy(self, inputs, ray_indices, occupancy_points, entropy_fn, 
                    type='ent', surface_points=None, assist_grid=None, scene_aabb=None, mean=True):
        if type== 'no_surface':
            surface_grids = occupancy_points
        else:
            # TODO fix the mask threshold 0.36->0.8            
            surface_mask = \
                torch.where(assist_grid.occs.reshape(assist_grid.binaries.shape)>0.8, 1.0, 0.0)
            # surface_points = nerfacc.grid.query_grid(inputs, self.grid.roi_aabb, 
            #                                         surface_mask, self.grid.contraction_type)
            # surface_mask shape : torch.Size([1, 128, 128, 128])
            if surface_points is None:
                surface_points, surface_pnt_mask = nerfacc.grid._query(inputs, surface_mask, 
                                                                   scene_aabb)
            ray_wsurface =set(ray_indices[surface_points>0].detach().cpu().numpy())
            ray_wosurface = set(ray_indices[surface_points==0].detach().cpu().numpy()) - ray_wsurface
            surface_grids = occupancy_points*surface_points
            # surface_grid_idx = surface_grids>0
            # surface_extract_grids = surface_grids[surface_grids>0]
        
        acquisition_value = entropy_fn(surface_grids)
        # acquisition_value = torch.mean(acquisition_value)
        acq_full = torch.where(surface_grids>0, acquisition_value, 0.0)
        if type=='ent' and len(ray_wosurface)>0:
            for ele in ray_wosurface:
                # occupied_idx = ray_indices==ele
                extra_acquisition_value = entropy_fn(occupancy_points)
                # extra_surface_grids = occupancy_points[ray_indices==ele]
                # extra_acquisition_value = entropy_fn(extra_surface_grids)
                # extra_acquisition_value = torch.mean(extra_acquisition_value)
                acq_extra = torch.where(ray_indices==ele, extra_acquisition_value, 0.0)
                acq_full = torch.sum(torch.stack([acq_full, acq_extra], 0), 0)
        if mean:
            acq_full = torch.mean(acq_full)
        # else:
        #     acq_full =  torch.where(acq_full!=0, acq_full, torch.log(torch.tensor(1e-6).cuda())/2 + 1/2)
        return acq_full, surface_points    
    
    @torch.no_grad()
    def eval_k_views(self, ray_bundle: RayBundle,
        render_step_size: float,
        bg_render_step_size: float,
        near_plane: float = 0.0,
        far_plane: Optional[float] = None,
        cone_angle: float = 0.0,
        alpha_sample_thre: float = 1e-2,
          type='ent', grid_sampling=True, bg_flag=False):
        # sampling from occupancy grids
        # ray_bundle shape (1024,)
        def make_entropy(val):
            # you can delete 1e-6 when the self.alpha_sample_thre is low enough.
            val = val.clip(1e-6, 1.0- (1e-6))
            return -val*torch.log(val) -(1-val)*torch.log(1-val)
        n_rays = ray_bundle.shape[0]
        if grid_sampling and self._update_counter.item() > 0:
            ray_samples, ray_indices = self(ray_bundle, render_step_size, near_plane, far_plane,
                                            0.0, 0.0)
            packed_info = nerfacc.pack_info(ray_indices, n_rays)
            if bg_flag:
                bg_ray_samples, bg_ray_indices = self(ray_bundle, bg_render_step_size, 
                                                        near_plane, far_plane,
                                            cone_angle, alpha_sample_thre, bg_model=True)
                bg_inputs = bg_ray_samples.frustums.get_start_positions()
                bg_inputs = bg_inputs.view(-1, 3)
        elif grid_sampling:
            ray_samples, ray_indices = self(ray_bundle, render_step_size, near_plane, far_plane,
                                            0.0, 0.0)
            n_rays, n_samples = ray_samples.shape
            ray_indices = torch.arange(n_rays)
            ray_indices = ray_indices.repeat_interleave(n_samples)
            packed_info = None
        else:
            ray_samples = self.uniform_sampler(ray_bundle)
            n_rays, n_samples = ray_samples.shape
            ray_indices = torch.arange(n_rays)
            ray_indices = ray_indices.repeat_interleave(n_samples)
            packed_info = None
        inputs = ray_samples.frustums.get_start_positions()
        # starts = ray_samples.frustums.starts
        self.n_rays = n_rays
        self.packed_info = packed_info
        inputs = inputs.view(-1, 3)
        # if grid_sampling and self.bg_flag:
        #     inputs = torch.cat([inputs, bg_inputs], dim=0)
        #     ray_indices = torch.cat([ray_indices, bg_ray_indices], dim=0)
        # 1 * (0.95)^16

        # occupancy_points = nerfacc.grid.query_grid(inputs, self.grid.roi_aabb, 
        #                                         self._alpha, self.grid.contraction_type)
        occupancy_points, occup_mask = nerfacc.grid._query(inputs, 
                                               self._alpha,
                                               self.scene_aabb)
        occupancy_acq, surface_points = self.cal_entropy(inputs, ray_indices, occupancy_points,
                                                          make_entropy, type, None,
                                                          self.assist_grid, self.scene_aabb)
        if grid_sampling and bg_flag:
            bg_occupancy_points, bg_occup_mask = nerfacc.grid._query(bg_inputs, 
                                               self.bg_alpha,
                                               self.bg_aabb)
            bg_occupancy_acq, surface_points = self.cal_entropy(bg_inputs, bg_ray_indices, bg_occupancy_points,
                                                          make_entropy, type, None,
                                                          self.bg_assist_grid, self.bg_aabb)
            occupancy_acq += bg_occupancy_acq
        
        
        return occupancy_acq, torch.mean(occupancy_points)
    
    @torch.no_grad()
    def initialize_grid(self, camera_info, chunk, near_plane):
        camera_to_worlds = camera_info['c2w']
        width = camera_info['width']
        height = camera_info['height']
        K = camera_info['K']
        self.assist_grid.mark_invisible_cells(K, camera_to_worlds, width, height, near_plane, chunk)
        if self.bg_flag:
            self.bg_assist_grid.mark_invisible_cells(K, camera_to_worlds, width, height, near_plane, chunk)
        # self.bg_grid.mark_invisible_cells(K, camera_to_worlds, width, height, near_plane, chunk)

    def evaluate(self, points, bg_flag = False):
        alphas = []
        for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
            
            # inside_sphere_mask = get_foreground_mask(ray_points=pnts)
            
            if not bg_flag:
                sdf = self.sdf_fn(pnts, steps=self.step)
                if sdf.ndim<2:
                    sdf = torch.unsqueeze(sdf, 1)
                estimated_next_sdf = sdf - self.step_size * 0.5
                estimated_prev_sdf = sdf + self.step_size * 0.5
                ### alpha
                inv_s = self.inv_s_fn()
                prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
                next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
                p = prev_cdf - next_cdf
                c = prev_cdf
                alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
                # alpha = alpha*inside_sphere_mask
            else:
            # if self.density_fn is not None:
                density = self.density_fn(pnts, steps=self.step)
                delta_density = self.step_size * density
                bg_alphas = 1 - torch.exp(-delta_density)
                # alpha = alpha*inside_sphere_mask + bg_alphas*(1-inside_sphere_mask)
                # alpha = bg_alphas*(1-inside_sphere_mask)
            alphas.append(alpha)
        alphas =torch.cat(alphas, axis=0)
        return alphas
    
    def update_density(self, points):
        return self.density_fn(points, steps=self.step)

    @torch.no_grad()
    def update_binary_grid(self, step):
        self.grid.update_every_n_steps(step=step, 
                                occ_eval_fn=self.alpha_pts, 
                                occ_thre=0.001,
                                ema_decay = self.decay,
                                warmup_steps = self.steps_warmup,
                                n = self.steps_per_grid_update,
                                )
        self.assist_grid.update_every_n_steps(step=step, 
                                occ_eval_fn=self.surface_pts, 
                                occ_thre=0.0,
                                ema_decay = self.decay,
                                warmup_steps = self.steps_warmup,
                                n = self.steps_per_grid_update,
                                )
        if self.density_fn is not None:
            self.bg_flag = True
            self.bg_grid.update_every_n_steps(step=step, 
                                    occ_eval_fn=self.update_density, 
                                    occ_thre=0.01,
                                    ema_decay = self.decay,
                                    warmup_steps = self.steps_warmup,
                                    n = self.steps_per_grid_update,
                                    )
            self.bg_assist_grid.update_every_n_steps(step=step, 
                                occ_eval_fn=self.surface_pts, 
                                occ_thre=0.0,
                                ema_decay = self.decay,
                                warmup_steps = self.steps_warmup,
                                n = self.steps_per_grid_update,
                                )

        if step % self.steps_per_grid_update == 0:
            flag = (self.assist_grid.occs<0).reshape(self.assist_grid.binaries.shape)
            alphas = self.evaluate(self.cube_coordinate)
            self._alpha = \
                    torch.where(flag,
                            self._alpha,
                            alphas.reshape(self.assist_grid.binaries.shape))
            
            # TODO bg merge
            if self.bg_flag:
                flag = (self.bg_assist_grid.occs<0).reshape(self.bg_assist_grid.binaries.shape)
                alphas = []
                for i in range(len(self.bg_cube_coordinate)):
                    alphas.append(self.evaluate(self.bg_cube_coordinate[i], bg_flag=True))
                alphas = torch.stack(alphas, dim=0)
                self.bg_alpha = \
                    torch.where(flag,
                            self.bg_alpha,
                            alphas.reshape(self.bg_assist_grid.binaries.shape)) 

        if step > self.steps_warmup and step % self.steps_per_grid_update == 0:
            self._update_counter += 1
        if step<(self.topk_iter*5-3) and step%self.topk_iter ==0:
            self._update_counter = torch.zeros(1, dtype=torch.int32)
            step = step%self.topk_iter

    def alpha_pts(self, points):
        sdf= self.sdf_fn(points, steps=self.step)
        # bound = self.voxel_size * (3**0.5) / 2.0

        estimated_next_sdf = sdf - self.step_size * 0.5
        estimated_prev_sdf = sdf + self.step_size * 0.5
        # import pdb
        # pdb.set_trace()
        inv_s = self.inv_s_fn()
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        return alpha #*self.render_stepsize
        
    def surface_pts(self, points):
        sdf= self.sdf_fn(points, steps=self.step)
        if sdf.ndim<2:
            sdf = torch.unsqueeze(sdf, 1)
        n_samples = sdf.shape[0]
        # bound = self.voxel_size * (3**0.5) / 2.0

        estimated_next_sdf = sdf - self.step_size * 0.5
        estimated_prev_sdf = sdf + self.step_size * 0.5
        # find surface
        sign_matrix = torch.sign(estimated_prev_sdf * estimated_next_sdf)
        sign_matrix = torch.cat(
            [sign_matrix, 
                torch.ones(n_samples, 1).to(sdf.device)], dim=-1
        )
        values, indices = torch.min(sign_matrix, -1)
        mask_sign_change = values < 0
        mask_pos_to_neg = estimated_prev_sdf[torch.arange(n_samples), 0] > 0
        # Define mask where a valid depth value is found
        mask = (mask_sign_change & mask_pos_to_neg).float()
        return mask

    def get_sigma_fn(self, origins, directions) -> Optional[Callable]:
        """Returns a function that returns the density of a point.
        """
        if self.density_fn is None or not self.training:
            return None
        density_fn = self.density_fn
        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = directions[ray_indices]
            # positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            outputs = density_fn(positions, steps=self.step)
            if type(outputs)==tuple:
                outputs=outputs[0]
            return torch.squeeze(outputs)

        return sigma_fn
    
    def get_alpha_fn(self, origins, directions) -> Optional[Callable]:
        """Returns a function that returns the density of a point.
        """
        if self.sdf_fn is None or not self.training:
            return None
        sdf_fn = self.sdf_fn
        inv_s_fn = self.inv_s_fn
        def alpha_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = directions[ray_indices]
            # positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sdf = sdf_fn(positions,steps=self.step)
            estimated_next_sdf = sdf - self.step_size * 0.5
            estimated_prev_sdf = sdf + self.step_size * 0.5
            inv_s = inv_s_fn()
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
            return torch.squeeze(alpha)

        return alpha_fn

    def forward(
        self,
        ray_bundle: RayBundle,
        render_step_size: float,
        near_plane: float = 0.0,
        far_plane: Optional[float] = None,
        cone_angle: float = 0.0,
        alpha_sample_thre: float = 1e-2,
        bg_model=False,
        sdf_fn=None,
    ) -> Tuple[RaySamples, TensorType["total_samples",]]:
        """Generate ray samples in a bounding box.
        """
        # self.pdb_debug = True
        if self.pdb_debug:
            pdb.set_trace()
        if bg_model and self._update_counter.item() <= 0:
            return self.uniform_sampler(ray_bundle)
        elif self._update_counter.item()<=0:
            if sdf_fn is None:
                sdf_fn = self.sdf_fn
            return self.neus_sampler(ray_bundle, sdf_fn=sdf_fn, 
                                        steps=self.step)

        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()
        times = ray_bundle.times
        if self.pdb_debug:
            pdb.set_trace()
        if ray_bundle.nears is not None and ray_bundle.fars is not None:
            t_min = ray_bundle.nears.contiguous().reshape(-1)
            t_max = ray_bundle.fars.contiguous().reshape(-1)
        else:
            t_min = None
            t_max = None
        
        # if bg_model:
        #     scene_aabb = self.aabb.reshape(1,6).to(ray_bundle.origins.device)
        #     t_min_aabb, t_max_aabb, hits = _ray_aabb_intersect(ray_bundle.origins, 
        #                                         ray_bundle.directions,
        #                                           aabbs=scene_aabb)
        #     t_min = t_max_aabb
        #     t_max = far_plane

        if self.pdb_debug:
            pdb.set_trace()

        if far_plane is None:
            far_plane = 1e10

        if ray_bundle.camera_indices is not None:
            camera_indices = ray_bundle.camera_indices.contiguous()
        else:
            camera_indices = None

        if render_step_size is None:
            if bg_model:
                render_step_size = 0.01
            else:
                render_step_size = max(self.step_size, 0.01)
        if self.pdb_debug:
            pdb.set_trace()

        if bg_model:
            ray_indices, starts, ends = self.bg_grid.sampling(
                rays_o=rays_o,
                rays_d=rays_d,
                t_min=t_min,
                t_max=t_max,
                sigma_fn=self.get_sigma_fn(rays_o, rays_d),
                render_step_size=render_step_size,
                near_plane=near_plane,
                far_plane=far_plane,
                stratified=self.training,
                cone_angle=cone_angle,
                alpha_thre=alpha_sample_thre,
            )
        else: 
            ray_indices, starts, ends = self.grid.sampling(
                rays_o=rays_o,
                rays_d=rays_d,
                t_min=t_min,
                t_max=t_max,
                alpha_fn=None,
                # sigma_fn=self.get_alpha_fn(rays_o, rays_d),
                render_step_size=render_step_size,
                near_plane=near_plane,
                far_plane=far_plane,
                stratified=self.training,
                cone_angle=0.0,
                alpha_thre=0.0,
            )
        num_samples = starts.shape[0]
        if self.pdb_debug:
            pdb.set_trace()
        if num_samples == 0:
            # print("OCC GRID SAMPLE 0!!!  Background model: ", bg_model)
            # create a single fake sample and update packed_info accordingly
            # this says the last ray in packed_info has 1 sample, which starts and ends at 1
            ray_indices = torch.zeros((1,), dtype=torch.long, device=rays_o.device)
            starts = torch.ones((1,), dtype=starts.dtype, device=rays_o.device)
            ends = torch.ones((1,), dtype=ends.dtype, device=rays_o.device)

        origins = rays_o[ray_indices]
        dirs = rays_d[ray_indices]
        if camera_indices is not None:
            camera_indices = camera_indices[ray_indices]
        if self.pdb_debug:
            pdb.set_trace()
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=dirs,
                starts=starts[..., None],
                ends=ends[..., None],
                pixel_area=ray_bundle[ray_indices].pixel_area,
            ),
            camera_indices=camera_indices,
            deltas = ends[..., None]-starts[...,None]
        )
        if ray_bundle.times is not None:
            ray_samples.times = ray_bundle.times[ray_indices]
        if self.pdb_debug:
            pdb.set_trace()
        return ray_samples, ray_indices

################################################
class ActiveNeuSAccSampler(NeuSAccSampler):
    """Voxel surface guided sampler in NeuralReconW."""

    def __init__(
        self,
        aabb,
        grid,
        bg_grid,
        assist_grid,
        bg_assist_grid=None,
        scene_aabb=None,
        bg_aabb=None,

        neus_sampler: NeuSSampler = None,
        resolution: int = 128,
        bg_resolution=64,

        steps_warmup: int = 256, #1200,
        steps_per_grid_update: int = 16, #600,
        coarse_sample: int = 64,
        topk_iter: int = 4000,

        density_fn=None,
        sdf_fn=None,
        inv_s_fn=None,
        pdb_debug=False,
    ) -> None:
        super().__init__(aabb=aabb, grid=grid, bg_grid=bg_grid, assist_grid=assist_grid,
                         bg_assist_grid=bg_assist_grid, scene_aabb=scene_aabb,
                         bg_aabb=bg_aabb, neus_sampler=neus_sampler, resolution=resolution,
                         bg_resolution=bg_resolution, steps_warmup=steps_warmup,
                         steps_per_grid_update=steps_per_grid_update,
                         coarse_sample=coarse_sample, topk_iter=topk_iter,
                         density_fn=density_fn,
                         sdf_fn=sdf_fn, inv_s_fn=inv_s_fn, pdb_debug=pdb_debug,)

        if density_fn is not None:
            self.bg_flag = True
            self.register_buffer("bg_uncertainty_binary", torch.ones(self.bg_grid.binaries.shape, dtype=torch.float))
        self.register_buffer("_uncertainty_binary", torch.ones(self.grid.binaries.shape, dtype=torch.float))

    
    @torch.no_grad()
    def eval_k_views(self, ray_bundle: RayBundle,
        render_step_size: float,
        bg_render_step_size: float,
        near_plane: float = 0.0,
        far_plane: Optional[float] = None,
        cone_angle: float = 0.0,
        alpha_sample_thre: float = 1e-2, type='ent', 
        kimera_type='none', grid_sampling=True, bg_flag=False, sdf_fn=None):
        # sampling from occupancy grids
        # ray_bundle shape (1024,)
        def make_gauss_entropy(val):
            # you can delete 1e-6 when the self.alpha_thres is low enough.
            return torch.log(2*np.pi*val+1e-6)/2 + 1/2
        
        def make_entropy(val):
            # you can delete 1e-6 when the self.alpha_thres is low enough.
            val = val.clip(1e-6, 1.0- (1e-6))
            return -val*torch.log(val) -(1-val)*torch.log(1-val)
        n_rays = ray_bundle.shape[0]
        # TODO how about neus_sampler?
        if grid_sampling and self._update_counter.item() > 0: 
            ray_samples, ray_indices = self(ray_bundle, render_step_size, near_plane,
                                             far_plane,
                                            0.0, 0.0, sdf_fn=sdf_fn)
            packed_info = nerfacc.pack_info(ray_indices, n_rays)
            if bg_flag:
                bg_ray_samples, bg_ray_indices = self(ray_bundle, bg_render_step_size, 
                                                        near_plane, far_plane,
                                            cone_angle, alpha_sample_thre, bg_model=True,
                                            sdf_fn=sdf_fn)
                bg_inputs = bg_ray_samples.frustums.get_start_positions()
                bg_inputs = bg_inputs.view(-1, 3)
        elif grid_sampling:
            ray_samples = self(ray_bundle, render_step_size, near_plane, far_plane,
                                            0.0, 0.0, sdf_fn=sdf_fn)
            n_rays, n_samples = ray_samples.shape
            ray_indices = torch.arange(n_rays)
            ray_indices = ray_indices.repeat_interleave(n_samples)
            packed_info = None
        else:
            ray_samples = self.uniform_sampler(ray_bundle)
            n_rays, n_samples = ray_samples.shape
            ray_indices = torch.arange(n_rays)
            ray_indices = ray_indices.repeat_interleave(n_samples)
            packed_info = None
        inputs = ray_samples.frustums.get_start_positions()
        # starts = ray_samples.frustums.starts
        self.n_rays = n_rays
        self.packed_info = packed_info
        inputs = inputs.view(-1, 3)
        # if grid_sampling and self.bg_flag:
        #     inputs = torch.cat([inputs, bg_inputs], dim=0)
        #     ray_indices = torch.cat([ray_indices, bg_ray_indices], dim=0)
        # 1 * (0.95)^16

        # uncertain_points = nerfacc.grid.query_grid(inputs, self.grid.roi_aabb, 
                                                # self._uncertainty_binary, self.grid.contraction_type)

        uncertain_points, uncert_pnt_mask = nerfacc.grid._query(inputs, 
                                               self._uncertainty_binary, 
                                               self.scene_aabb)
        uncertain_acq, surface_points = self.cal_entropy(inputs, ray_indices, uncertain_points,
                                                          make_gauss_entropy, type, None,
                                                          self.assist_grid, self.scene_aabb)
        if grid_sampling and bg_flag:
            bg_uncertain_points, uncert_pnt_mask = nerfacc.grid._query(bg_inputs, 
                                               self.bg_uncertainty_binary, 
                                               self.bg_aabb)
            bg_uncertain_acq, bg_surface_points = self.cal_entropy(bg_inputs, bg_ray_indices,
                                                                    bg_uncertain_points,
                                                          make_gauss_entropy, type, None,
                                                          self.bg_assist_grid, self.bg_aabb)
            uncertain_acq += bg_uncertain_acq
        if kimera_type=='entropy':
            occupancy_points, occup_mask = nerfacc.grid._query(inputs, 
                                               self._alpha, 
                                               self.scene_aabb)

            occupancy_acq, surface_points = self.cal_entropy(inputs, ray_indices, occupancy_points,
                                                            make_entropy, type, surface_points,
                                                            self.assist_grid, self.scene_aabb)
            if grid_sampling and bg_flag:
                bg_occupancy_points, bg_occup_mask = nerfacc.grid._query(bg_inputs, 
                                                self.bg_alpha,
                                                self.bg_aabb)
                bg_occupancy_acq, bg_surface_points = self.cal_entropy(bg_inputs, bg_ray_indices, 
                                                                    bg_occupancy_points,
                                                            make_entropy, type, bg_surface_points,
                                                            self.bg_assist_grid, self.bg_aabb)
                occupancy_acq += bg_occupancy_acq
            
            uncertain_acq += occupancy_acq
            # import pdb
            # pdb.set_trace()
        
        return uncertain_acq, torch.mean(uncertain_points)
    
    def evaluate(self, points, bg_flag=False):
        u, alphas = [], []
        
        for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
            
            # inside_sphere_mask = get_foreground_mask(ray_points=pnts)
            
            if not bg_flag:
                sdf, uncertainty = self.sdf_fn(pnts, return_sdf=True, steps=self.step)
                estimated_next_sdf = sdf - self.step_size * 0.5
                estimated_prev_sdf = sdf + self.step_size * 0.5
                ### alpha
                inv_s = self.inv_s_fn()
                prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
                next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
                p = prev_cdf - next_cdf
                c = prev_cdf
                alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
                # uncertainty = uncertainty*inside_sphere_mask
                # alpha = alpha*inside_sphere_mask
            else:
            # if self.density_fn is not None:
                density, bg_uncertainty, _ = self.density_fn(pnts, steps=self.step)
                delta_density = self.step_size * density
                bg_alphas = 1 - torch.exp(-delta_density)
                # uncertainty = uncertainty*inside_sphere_mask + bg_uncertainty*(1-inside_sphere_mask)
                # alpha = alpha*inside_sphere_mask + bg_alphas*(1-inside_sphere_mask)
                # uncertainty = bg_uncertainty*(1-inside_sphere_mask)
                # alpha = bg_alphas*(1-inside_sphere_mask)
            u.append(uncertainty)
            alphas.append(alpha)
        u = torch.cat(u, axis=0)
        alphas =torch.cat(alphas, axis=0)
        return u, alphas
    
    def density_pts(self, points):
        density, uncertainty, _= self.density_fn(points, steps=self.step)
        return density

    @torch.no_grad()
    def update_binary_grid(self, step):
        self.grid.update_every_n_steps(step=step, 
                                occ_eval_fn=self.alpha_pts, 
                                occ_thre=0.0009,
                                ema_decay = self.decay,
                                warmup_steps = self.steps_warmup,
                                n = self.steps_per_grid_update,
                                )
        self.assist_grid.update_every_n_steps(step=step, 
                                occ_eval_fn=self.surface_pts, 
                                occ_thre=0.0,
                                ema_decay = self.decay,
                                warmup_steps = self.steps_warmup,
                                n = self.steps_per_grid_update,
                                )
        if self.density_fn is not None:
            self.bg_flag = True
            self.bg_grid.update_every_n_steps(step=step, 
                                    occ_eval_fn=self.density_pts, 
                                    # lowering threshold because of uncertainty
                                    occ_thre=0.005,
                                    ema_decay = self.decay,
                                    warmup_steps = self.steps_warmup,
                                    n = self.steps_per_grid_update,
                                    )
            self.bg_assist_grid.update_every_n_steps(step=step, 
                                occ_eval_fn=self.surface_pts, 
                                occ_thre=0.0,
                                ema_decay = self.decay,
                                warmup_steps = self.steps_warmup,
                                n = self.steps_per_grid_update,
                                )

        # self._binary = self.grid.binary
        if step % self.steps_per_grid_update == 0:
            flag = (self.assist_grid.occs<0).reshape(self.assist_grid.binaries.shape)
            uncertainty, alphas = self.evaluate(self.cube_coordinate)
            if self.pdb_debug:
                pdb.set_trace()
            self._alpha = \
                torch.where(flag,
                        self._alpha,
                        alphas.reshape(self.assist_grid.binaries.shape))

            self._uncertainty_binary = \
                torch.where(flag,
                        self._uncertainty_binary,
                        torch.minimum(self._uncertainty_binary*(2-self.decay), 
                                        uncertainty.reshape(self.assist_grid.binaries.shape)))
            if self.pdb_debug:
                pdb.set_trace()
            if self.bg_flag:
                flag = (self.bg_assist_grid.occs<0).reshape(self.bg_assist_grid.binaries.shape)
                uncertainty, alphas = [], []
                for i in range(len(self.bg_cube_coordinate)):
                    uncert, alpha = self.evaluate(self.bg_cube_coordinate[i], bg_flag=True)
                    alphas.append(alpha)
                    uncertainty.append(uncert)
                alphas = torch.stack(alphas, dim=0)
                uncertainty = torch.stack(uncertainty, dim=0)
                if self.pdb_debug:
                    pdb.set_trace()
                self.bg_alpha = \
                    torch.where(flag,
                            self.bg_alpha,
                            alphas.reshape(self.bg_assist_grid.binaries.shape)) 
                self.bg_uncertainty_binary = \
                    torch.where(flag,
                            self.bg_uncertainty_binary,
                            torch.minimum(self.bg_uncertainty_binary*(2-self.decay), 
                                            uncertainty.reshape(self.bg_assist_grid.binaries.shape)))
                if self.pdb_debug:
                    pdb.set_trace()
                
             
        if step > self.steps_warmup and step % self.steps_per_grid_update == 0:
            self._update_counter += 1
        if step<(self.topk_iter*5-3) and step%self.topk_iter ==0:
            # Use NeuS sampler again
            self._update_counter = torch.zeros(1, dtype=torch.int32)
            step = step%self.topk_iter
        if self.pdb_debug:
            pdb.set_trace()
    
########################################################