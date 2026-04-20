# nerf-mini

A minimal implementation built from scratch in pure PyTorch — no CUDA required.

This project was built as a portfolio piece for a PhD application in Computer Graphics / Neural Rendering at Lund University. It demonstrates the core ideas behind neural rendering: representing a 3D scene as a continuous function learned by a neural network, then rendering novel views via volumetric ray marching.

---

## What is NeRF?

A Neural Radiance Field represents a 3D scene as a function:

```
f(x, y, z, θ, φ) → (R, G, B, σ)
```

Where:
- `(x, y, z)` is a 3D point in space
- `(θ, φ)` is the viewing direction
- `(R, G, B)` is the emitted colour
- `σ` is the volume density (how opaque the point is)

This function is approximated by a small MLP (Multi-Layer Perceptron). To render a pixel, we cast a ray through the scene, sample points along it, query the MLP at each point, and composite the results using the **volume rendering equation**:

```
C(r) = ∫ T(t) · σ(r(t)) · c(r(t), d) dt
```

where `T(t) = exp(-∫₀ᵗ σ(r(s)) ds)` is the transmittance (how much light reaches that point).

---

## Project Structure

```
nerf-mini/
├── src/
│   ├── nerf_model.py    # TinyNeRF MLP + positional encoding
│   ├── ray_march.py     # ray generation, sampling, volume rendering
│   └── dataset.py       # synthetic cube scene + camera poses
├── train.py             # training loop with logging + validation
├── viewer.py            # renders turntable GIF from trained model
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

PyTorch CPU-only (smaller download):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Train

```bash
python train.py
```

Quick test (fewer iterations):
```bash
python train.py --iters 1000 --log_every 100 --val_every 200
```

Higher quality (takes longer):
```bash
python train.py --iters 10000 --height 100 --width 100
```

### 3. Render turntable GIF

```bash
python viewer.py
```

Output: `results/turntable.gif`

---

## Key Components

### `src/nerf_model.py` — TinyNeRF

The MLP takes positionally-encoded 3D coordinates and view directions as input. **Positional encoding** maps raw coordinates to a higher-dimensional space of sinusoids, which is essential for the network to learn high-frequency detail:

```
γ(x) = [sin(2⁰πx), cos(2⁰πx), sin(2¹πx), cos(2¹πx), ..., sin(2^(L-1)πx), cos(2^(L-1)πx)]
```

Architecture:
- Geometry branch: `pos_enc → 4×Linear(128) → density + feature`
- Appearance branch: `feature + dir_enc → 2×Linear → RGB`

### `src/ray_march.py` — Volumetric Rendering

1. **Ray generation**: compute ray origin + direction for each pixel using camera intrinsics and the camera-to-world pose matrix
2. **Stratified sampling**: divide each ray into N bins, sample one point per bin (with random jitter during training for better coverage)
3. **Volume compositing**: integrate colour and density using the discrete approximation of the rendering integral

### `src/dataset.py` — Synthetic Scene

Generates a procedural coloured cube with known camera poses arranged in a circle. No external data needed — the scene is rendered analytically for ground-truth training images.

---

## Results

After ~5000 iterations on the synthetic scene:

| Metric | Value |
|--------|-------|
| Val PSNR | ~25–30 dB |
| Training time (CPU) | ~10–20 min |
| Model parameters | ~200K |

Training outputs:
- `results/loss_curve.png` — training loss + validation PSNR curves
- `results/val_render_iter_*.png` — validation renders at each checkpoint
- `results/turntable.gif` — novel-view synthesis animation
- `results/metrics.txt` — full PSNR log

---

## Concepts Demonstrated

| Concept | Where |
|---------|-------|
| Neural implicit representation | `nerf_model.py` — MLP as scene function |
| Positional encoding (Fourier features) | `nerf_model.py` — `PositionalEncoding` class |
| Volumetric ray marching | `ray_march.py` — `render_rays()` |
| Volume rendering equation | `ray_march.py` — `volume_render()` |
| Camera model (pinhole) | `ray_march.py` — `get_rays()` |
| 3D scene representation | `dataset.py` — synthetic cube + orbit cameras |
| GPU-ready training loop | `train.py` — works on CPU; move to GPU with `model.cuda()` |

---

## Extending This Project

Some directions to explore next:

- **Hierarchical sampling** (coarse + fine networks) — as in the original NeRF paper
- **Instant-NGP style hash encoding** — for much faster training
- **CUDA kernel** for ray marching — the bottleneck on CPU
- **Real scene data** — try the [NeRF synthetic dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) (Blender scenes)
- **Gaussian Splatting** — a newer, faster alternative to NeRF

---

## References

1. Mildenhall et al. (2020). *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.* ECCV 2020. https://arxiv.org/abs/2003.08934
2. Tancik et al. (2020). *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.* NeurIPS 2020.
3. Müller et al. (2022). *Instant Neural Graphics Primitives.* SIGGRAPH 2022.

---

## License

MIT License — free to use, modify, and distribute.
