"""
viewer.py
---------
Renders a turntable animation of a trained NeRF model and saves it as a GIF.

Usage:
    python viewer.py                              # uses checkpoints/nerf_final.pt
    python viewer.py --ckpt checkpoints/nerf_best.pt
    python viewer.py --frames 60 --height 128 --width 128

Output:
    results/turntable.gif
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from nerf_model import TinyNeRF
from ray_march  import get_rays, render_rays
from dataset    import orbit_cameras

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def render_frame(model, c2w, height, width, focal, near, far, num_samples):
    model.eval()
    origins, dirs = get_rays(height, width, focal, c2w)
    flat_o = origins.reshape(-1, 3)
    flat_d = dirs.reshape(-1, 3)
    with torch.no_grad():
        rgb = render_rays(model, flat_o, flat_d,
                          near=near, far=far,
                          num_samples=num_samples, perturb=False)
    return rgb.reshape(height, width, 3).clamp(0, 1)


def main(args):
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        print("Please run train.py first.")
        return

    print(f"Loading checkpoint: {args.ckpt}")
    model = TinyNeRF(pos_freq=6, dir_freq=4, hidden_dim=128)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    poses = orbit_cameras(args.frames, radius=4.0, elevation=args.elevation)

    print(f"Rendering {args.frames} frames at {args.height}×{args.width}...")
    frames = []
    for i, c2w in enumerate(poses):
        frame = render_frame(model, c2w, args.height, args.width,
                             args.focal, args.near, args.far, args.num_samples)
        frames.append(frame)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.frames} frames rendered")

    # Save as GIF using PIL
    if HAS_PIL:
        pil_frames = []
        for f in frames:
            arr = (f.numpy() * 255).astype("uint8")
            pil_frames.append(Image.fromarray(arr))

        out_path = "results/turntable.gif"
        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=50,    # ms per frame
            loop=0,
        )
        print(f"Saved turntable GIF: {out_path}")
    else:
        # Fallback: save individual PNG frames
        if HAS_MPL:
            for i, frame in enumerate(frames):
                path = f"results/frame_{i:03d}.png"
                plt.imsave(path, frame.numpy())
            print(f"Saved {len(frames)} PNG frames to results/")
        else:
            print("Install Pillow or matplotlib to save output: pip install pillow matplotlib")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render turntable GIF from trained NeRF")
    p.add_argument("--ckpt",        type=str,   default="checkpoints/nerf_final.pt")
    p.add_argument("--frames",      type=int,   default=36)
    p.add_argument("--height",      type=int,   default=64)
    p.add_argument("--width",       type=int,   default=64)
    p.add_argument("--focal",       type=float, default=50.0)
    p.add_argument("--near",        type=float, default=2.0)
    p.add_argument("--far",         type=float, default=6.0)
    p.add_argument("--num_samples", type=int,   default=64)
    p.add_argument("--elevation",   type=float, default=30.0)
    args = p.parse_args()
    main(args)
