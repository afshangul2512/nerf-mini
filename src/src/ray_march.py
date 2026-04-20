"""
ray_march.py
------------
Volumetric ray marching for NeRF.

Given camera rays, we:
  1. Sample 3D points along each ray
  2. Query the NeRF MLP at each point
  3. Composite colours using the volume rendering equation

The math: C(r) = ∫ T(t) · σ(r(t)) · c(r(t), d) dt
where T(t) = exp(-∫₀ᵗ σ(r(s)) ds)   (transmittance)

We approximate this integral with discrete samples.
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Camera ray generation
# ---------------------------------------------------------------------------

def get_rays(
    height: int,
    width: int,
    focal: float,
    camera_to_world: torch.Tensor,  # (4, 4) camera pose matrix
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate one ray per pixel in world space.

    Returns:
        origins:    (H, W, 3) — all rays start from the camera centre
        directions: (H, W, 3) — unit direction vectors
    """
    # Pixel grid in image (camera) space
    i, j = torch.meshgrid(
        torch.arange(width,  dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing="xy",
    )
    # Convert pixel coords to camera-space directions
    dirs = torch.stack([
        (i - width  * 0.5) / focal,
        -(j - height * 0.5) / focal,   # y flipped (image vs. world convention)
        -torch.ones_like(i),            # looking down -Z
    ], dim=-1)   # (H, W, 3)

    # Rotate directions into world space using the rotation part of c2w
    R = camera_to_world[:3, :3]          # (3, 3)
    directions = (dirs @ R.T)            # (H, W, 3)
    directions = F.normalize(directions, dim=-1)

    # Ray origins: camera centre in world space (translation column)
    origins = camera_to_world[:3, 3].expand_as(directions)  # (H, W, 3)

    return origins, directions


# ---------------------------------------------------------------------------
# Point sampling along rays
# ---------------------------------------------------------------------------

def sample_points_along_rays(
    origins: torch.Tensor,     # (..., 3)
    directions: torch.Tensor,  # (..., 3)
    near: float = 2.0,
    far:  float = 6.0,
    num_samples: int = 64,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample evenly-spaced (+ optional random jitter) points along each ray.

    Returns:
        points: (..., num_samples, 3)
        t_vals: (..., num_samples)       depth values
    """
    t_vals = torch.linspace(near, far, num_samples, device=origins.device)

    if perturb:
        # Stratified sampling: add uniform noise within each bin
        mid = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mid, t_vals[..., -1:]], dim=-1)
        lower = torch.cat([t_vals[..., :1], mid], dim=-1)
        t_rand = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * t_rand

    # r(t) = origin + t * direction
    points = origins[..., None, :] + t_vals[..., None] * directions[..., None, :]
    return points, t_vals


# ---------------------------------------------------------------------------
# Volume rendering (compositing)
# ---------------------------------------------------------------------------

def volume_render(
    rgb: torch.Tensor,       # (..., num_samples, 3)
    density: torch.Tensor,   # (..., num_samples)
    t_vals: torch.Tensor,    # (..., num_samples)
    white_background: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Composite samples into a single pixel colour using the volume rendering eq.

    Returns:
        pixel_rgb:    (..., 3)
        weights:      (..., num_samples)  — useful for depth maps
    """
    # Distances between consecutive samples
    deltas = t_vals[..., 1:] - t_vals[..., :-1]
    # Append a large value for the last segment
    deltas = torch.cat([deltas, torch.full_like(deltas[..., :1], 1e10)], dim=-1)

    # Alpha = 1 - exp(-σ·δ)
    alpha = 1.0 - torch.exp(-density * deltas)

    # Transmittance T(t) = ∏_{i<t} (1 - alpha_i)
    # Computed as exclusive cumprod
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[..., :-1]

    weights = transmittance * alpha    # (..., num_samples)

    # Composite colour
    pixel_rgb = (weights[..., None] * rgb).sum(dim=-2)   # (..., 3)

    if white_background:
        # Fill transparent regions with white
        acc = weights.sum(dim=-1, keepdim=True)
        pixel_rgb = pixel_rgb + (1.0 - acc)

    return pixel_rgb, weights


# ---------------------------------------------------------------------------
# Full render pass (rays → pixels)
# ---------------------------------------------------------------------------

def render_rays(
    model,
    origins: torch.Tensor,     # (N_rays, 3)
    directions: torch.Tensor,  # (N_rays, 3)
    near: float = 2.0,
    far:  float = 6.0,
    num_samples: int = 64,
    perturb: bool = True,
    chunk: int = 4096,         # process this many samples at once (memory limit)
) -> torch.Tensor:
    """
    Full forward pass: cast rays → sample points → query MLP → composite.

    Returns:
        pixel_rgb: (N_rays, 3)
    """
    points, t_vals = sample_points_along_rays(
        origins, directions, near=near, far=far,
        num_samples=num_samples, perturb=perturb,
    )
    # (N_rays, num_samples, 3) → flatten to (N_rays * num_samples, 3)
    N_rays = origins.shape[0]
    flat_pts  = points.reshape(-1, 3)
    flat_dirs = directions[:, None, :].expand_as(points).reshape(-1, 3)

    # Query MLP in chunks to avoid OOM on CPU
    all_rgb, all_density = [], []
    for i in range(0, flat_pts.shape[0], chunk):
        pts_chunk  = flat_pts[i:i+chunk]
        dirs_chunk = flat_dirs[i:i+chunk]
        with torch.no_grad() if not model.training else torch.enable_grad():
            rgb_c, den_c = model(pts_chunk, dirs_chunk)
        all_rgb.append(rgb_c)
        all_density.append(den_c)

    rgb     = torch.cat(all_rgb,     dim=0).reshape(N_rays, num_samples, 3)
    density = torch.cat(all_density, dim=0).reshape(N_rays, num_samples)

    pixel_rgb, _ = volume_render(rgb, density, t_vals, white_background=True)
    return pixel_rgb
