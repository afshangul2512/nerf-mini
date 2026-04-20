"""
dataset.py
----------
Generates a simple synthetic scene (a coloured cube) with known camera poses.
No real data download needed — everything is procedurally created.

This lets the project run completely offline and out-of-the-box.
"""

import torch
import math


# ---------------------------------------------------------------------------
# Camera pose helpers
# ---------------------------------------------------------------------------

def look_at(
    eye: torch.Tensor,     # (3,) camera position
    target: torch.Tensor,  # (3,) point to look at
    up: torch.Tensor,      # (3,) world up vector
) -> torch.Tensor:
    """Build a 4×4 camera-to-world matrix."""
    z_axis = torch.nn.functional.normalize(eye - target, dim=0)  # forward
    x_axis = torch.nn.functional.normalize(torch.cross(up, z_axis), dim=0)
    y_axis = torch.cross(z_axis, x_axis)

    c2w = torch.eye(4)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = eye
    return c2w


def orbit_cameras(
    num_views: int,
    radius: float = 4.0,
    elevation: float = 30.0,
) -> list[torch.Tensor]:
    """
    Place cameras uniformly around the Y-axis at a given elevation.
    Returns list of (4, 4) camera-to-world matrices.
    """
    poses = []
    elev_rad = math.radians(elevation)
    target = torch.zeros(3)
    up = torch.tensor([0.0, 1.0, 0.0])

    for i in range(num_views):
        azimuth = 2 * math.pi * i / num_views
        x = radius * math.cos(elev_rad) * math.cos(azimuth)
        y = radius * math.sin(elev_rad)
        z = radius * math.cos(elev_rad) * math.sin(azimuth)
        eye = torch.tensor([x, y, z])
        poses.append(look_at(eye, target, up))

    return poses


# ---------------------------------------------------------------------------
# Synthetic scene: a coloured unit cube
# ---------------------------------------------------------------------------

def cube_density_and_color(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Analytic ground-truth for a unit cube centred at origin.
    Inside the cube: high density + face-dependent colour.
    Outside: density ≈ 0.

    Args:
        points: (N, 3)
    Returns:
        density: (N,)
        rgb:     (N, 3)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    inside = (x.abs() < 0.5) & (y.abs() < 0.5) & (z.abs() < 0.5)

    density = torch.where(inside, torch.full_like(x, 100.0), torch.zeros_like(x))

    # Colour by face: map (x,y,z) from [-0.5, 0.5] to [0, 1]
    rgb = (points + 0.5).clamp(0, 1)   # (N, 3) — nice gradient across faces

    return density, rgb


# ---------------------------------------------------------------------------
# Render synthetic training images
# ---------------------------------------------------------------------------

def render_synthetic_image(
    c2w: torch.Tensor,
    height: int,
    width: int,
    focal: float,
    near: float = 2.0,
    far:  float = 6.0,
    num_samples: int = 64,
) -> torch.Tensor:
    """
    Ray-march through the analytic cube scene to produce a ground-truth image.
    Returns: (H, W, 3) float tensor in [0, 1].
    """
    from ray_march import get_rays, sample_points_along_rays, volume_render

    origins, directions = get_rays(height, width, focal, c2w)
    flat_o = origins.reshape(-1, 3)
    flat_d = directions.reshape(-1, 3)

    points, t_vals = sample_points_along_rays(
        flat_o, flat_d, near=near, far=far,
        num_samples=num_samples, perturb=False,
    )
    flat_pts = points.reshape(-1, 3)
    density, rgb = cube_density_and_color(flat_pts)

    density = density.reshape(-1, num_samples)
    rgb     = rgb.reshape(-1, num_samples, 3)

    pixel_rgb, _ = volume_render(rgb, density, t_vals, white_background=True)
    return pixel_rgb.reshape(height, width, 3)


# ---------------------------------------------------------------------------
# Build full dataset
# ---------------------------------------------------------------------------

def build_dataset(
    num_train: int = 40,
    num_val: int   = 8,
    height: int    = 64,
    width: int     = 64,
    focal: float   = 50.0,
) -> dict:
    """
    Generate synthetic training + validation splits.
    Returns a dict with keys: train_images, train_poses, val_images, val_poses,
                              height, width, focal.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    all_poses = orbit_cameras(num_train + num_val, radius=4.0, elevation=30.0)
    train_poses = all_poses[:num_train]
    val_poses   = all_poses[num_train:]

    print(f"Rendering {num_train} training images ({height}×{width})...")
    train_images = torch.stack([
        render_synthetic_image(p, height, width, focal) for p in train_poses
    ])

    print(f"Rendering {num_val} validation images...")
    val_images = torch.stack([
        render_synthetic_image(p, height, width, focal) for p in val_poses
    ])

    return {
        "train_images": train_images,   # (N_train, H, W, 3)
        "train_poses":  torch.stack(train_poses),
        "val_images":   val_images,
        "val_poses":    torch.stack(val_poses),
        "height": height,
        "width":  width,
        "focal":  focal,
    }
