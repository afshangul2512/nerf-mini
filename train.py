"""
train.py
--------
Train a TinyNeRF model on the synthetic cube scene.

Usage:
    python train.py                     # default settings
    python train.py --iters 2000        # quick test
    python train.py --height 64 --width 64 --iters 5000

Outputs:
    results/loss_curve.png
    results/val_render_iter_XXXX.png
    results/metrics.txt
    checkpoints/nerf_final.pt
"""

import argparse
import os
import sys
import time

import torch
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from nerf_model import TinyNeRF
from ray_march  import get_rays, render_rays
from dataset    import build_dataset

# Optional: matplotlib for saving plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def psnr(mse: float) -> float:
    """Peak Signal-to-Noise Ratio from mean squared error."""
    import math
    return -10.0 * math.log10(max(mse, 1e-10))


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save (H, W, 3) float tensor as PNG."""
    if not HAS_MPL:
        return
    img = tensor.clamp(0, 1).detach().cpu().numpy()
    plt.imsave(path, img)


def render_full_image(model, c2w, height, width, focal, near, far, num_samples):
    """Render a full image without computing gradients."""
    model.eval()
    origins, directions = get_rays(height, width, focal, c2w)
    flat_o = origins.reshape(-1, 3)
    flat_d = directions.reshape(-1, 3)

    with torch.no_grad():
        pixel_rgb = render_rays(
            model, flat_o, flat_d,
            near=near, far=far,
            num_samples=num_samples,
            perturb=False,
        )
    model.train()
    return pixel_rgb.reshape(height, width, 3)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs("results",     exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # --- Data ---
    print("Building synthetic dataset...")
    data = build_dataset(
        num_train=args.num_train,
        num_val=args.num_val,
        height=args.height,
        width=args.width,
        focal=args.focal,
    )
    train_images = data["train_images"]   # (N, H, W, 3)
    train_poses  = data["train_poses"]    # (N, 4, 4)
    val_images   = data["val_images"]
    val_poses    = data["val_poses"]
    H, W, f      = data["height"], data["width"], data["focal"]

    print(f"Dataset: {len(train_images)} train, {len(val_images)} val  |  {H}×{W} px")

    # --- Model & optimiser ---
    model = TinyNeRF(pos_freq=6, dir_freq=4, hidden_dim=args.hidden_dim)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9998)

    # --- Training ---
    losses     = []
    val_psnrs  = []
    best_psnr  = -float("inf")
    t0         = time.time()

    print(f"\nTraining for {args.iters} iterations...\n")

    for it in range(1, args.iters + 1):
        # Pick a random training image
        idx   = torch.randint(0, len(train_images), (1,)).item()
        c2w   = train_poses[idx]
        image = train_images[idx]    # (H, W, 3)

        # Sample random pixel rays
        coords = torch.stack(
            torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"),
            dim=-1,
        ).reshape(-1, 2)
        sel = torch.randperm(coords.shape[0])[:args.ray_batch]
        coords = coords[sel]

        # Get rays for selected pixels
        origins_all, dirs_all = get_rays(H, W, f, c2w)
        origins = origins_all[coords[:, 0], coords[:, 1]]
        dirs    = dirs_all   [coords[:, 0], coords[:, 1]]
        target  = image      [coords[:, 0], coords[:, 1]]   # (ray_batch, 3)

        # Forward pass
        optimizer.zero_grad()
        pred = render_rays(
            model, origins, dirs,
            near=args.near, far=args.far,
            num_samples=args.num_samples,
            perturb=True,
            chunk=args.chunk,
        )
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        # --- Logging ---
        if it % args.log_every == 0 or it == 1:
            elapsed = time.time() - t0
            print(f"  iter {it:5d}/{args.iters}  |  loss {loss.item():.4f}  "
                  f"|  PSNR {psnr(loss.item()):6.2f} dB  |  {elapsed:.1f}s elapsed")

        # --- Validation render ---
        if it % args.val_every == 0 or it == args.iters:
            val_idx  = 0
            val_pred = render_full_image(
                model, val_poses[val_idx], H, W, f,
                args.near, args.far, args.num_samples,
            )
            val_gt  = val_images[val_idx]
            val_mse = torch.mean((val_pred - val_gt) ** 2).item()
            vp      = psnr(val_mse)
            val_psnrs.append((it, vp))

            out_path = f"results/val_render_iter_{it:05d}.png"
            save_image(val_pred, out_path)
            print(f"  >>> Val PSNR {vp:.2f} dB — saved {out_path}")

            if vp > best_psnr:
                best_psnr = vp
                torch.save(model.state_dict(), "checkpoints/nerf_best.pt")

    # --- Save final model ---
    torch.save(model.state_dict(), "checkpoints/nerf_final.pt")

    # --- Save metrics ---
    with open("results/metrics.txt", "w") as f_out:
        f_out.write(f"Total iterations: {args.iters}\n")
        f_out.write(f"Best val PSNR:    {best_psnr:.2f} dB\n\n")
        f_out.write("Validation PSNR log:\n")
        for it, vp in val_psnrs:
            f_out.write(f"  iter {it:6d}: {vp:.2f} dB\n")
    print("\nSaved results/metrics.txt")

    # --- Loss curve ---
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(losses, alpha=0.7, linewidth=0.8)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("MSE Loss")
        axes[0].set_title("Training Loss")
        axes[0].set_yscale("log")

        iters_vp, psnrs_vp = zip(*val_psnrs) if val_psnrs else ([], [])
        axes[1].plot(iters_vp, psnrs_vp, marker="o")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("PSNR (dB)")
        axes[1].set_title("Validation PSNR")

        plt.tight_layout()
        plt.savefig("results/loss_curve.png", dpi=120)
        print("Saved results/loss_curve.png")

    print(f"\nDone! Best val PSNR: {best_psnr:.2f} dB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train TinyNeRF on synthetic cube scene")
    p.add_argument("--iters",       type=int,   default=5000,  help="training iterations")
    p.add_argument("--height",      type=int,   default=64)
    p.add_argument("--width",       type=int,   default=64)
    p.add_argument("--focal",       type=float, default=50.0)
    p.add_argument("--near",        type=float, default=2.0)
    p.add_argument("--far",         type=float, default=6.0)
    p.add_argument("--num_samples", type=int,   default=64,    help="samples per ray")
    p.add_argument("--ray_batch",   type=int,   default=512,   help="rays per iteration")
    p.add_argument("--chunk",       type=int,   default=4096,  help="MLP chunk size")
    p.add_argument("--hidden_dim",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--num_train",   type=int,   default=40)
    p.add_argument("--num_val",     type=int,   default=8)
    p.add_argument("--log_every",   type=int,   default=100)
    p.add_argument("--val_every",   type=int,   default=500)
    args = p.parse_args()
    train(args)
