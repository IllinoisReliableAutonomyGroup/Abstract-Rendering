import os
import sys
import yaml

import matplotlib
matplotlib.use("Agg")  # No GUI popups on SSH
import matplotlib.pyplot as plt
import argparse

import numpy as np
import torch
from matplotlib import cm
from scipy.ndimage import gaussian_filter


# Add repo root to PYTHONPATH (as advisor requested)
grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
if grandfather_path not in sys.path:
    sys.path.append(grandfather_path)


def _to_hwc(img):
    """Convert torch tensor (CHW or HWC) to numpy HWC [0,1]."""
    if isinstance(img, torch.Tensor):
        img = img.float().cpu()
    else:
        img = torch.as_tensor(img).float()

    if img.ndim == 3 and img.shape[0] == 3:
        img = img.permute(1, 2, 0)

    arr = img.numpy()
    if arr.max() > 1.01:
        arr = arr / 255.0
    return np.clip(arr, 0, 1)


def _load_ref(abstract_dir, idx, shape):
    """Load ref_{idx:06d}.png or return gray placeholder."""
    from PIL import Image

    ref_path = os.path.join(abstract_dir, f"ref_{idx:06d}.png")
    h, w, _ = shape

    if not os.path.exists(ref_path):
        return np.ones(shape, dtype=np.float32) * 0.5

    ref = Image.open(ref_path).convert("RGB")
    ref = np.asarray(ref).astype(np.float32) / 255.0

    # Resize to match lower/upper
    if ref.shape[:2] != (h, w):
        ref = np.array(ref.resize((w, h)))

    return np.clip(ref, 0, 1)


def visualize_all(abstract_dir, out=None):
    """
    Visualize all available indices in the given folder.
    """
    # Find all abstract records in the folder
    abstract_files = [
        f for f in os.listdir(abstract_dir) if f.startswith("abstract_") and f.endswith(".pt")
    ]
    indices = sorted(int(f.split("_")[1].split(".")[0]) for f in abstract_files)

    # Visualize each index
    for idx in indices:
        visualize(abstract_dir, idx, out)


def visualize(abstract_dir, idx=0, out=None):
    """
    Create a 2x2 panel:
      [ ref | lower ]
      [ upper | blended heatmap ]
    """

    rec_path = os.path.join(abstract_dir, f"abstract_{idx:06d}.pt")
    rec = torch.load(rec_path, map_location="cpu")

    lower = _to_hwc(rec["lower"])
    upper = _to_hwc(rec["upper"])

    h, w, _ = lower.shape

    ref = _load_ref(abstract_dir, idx, lower.shape)

    # Compute pixelwise uncertainty
    diff = np.abs(upper - lower).mean(axis=2)
    diff = gaussian_filter(diff, sigma=1.0)

    # Normalize to 0..1
    dmax = np.percentile(diff, 99)
    diff_norm = np.clip(diff / (dmax if dmax > 0 else 1), 0, 1)

    # Sky color = median of ref
    sky_color = np.median(ref.reshape(-1, 3), axis=0)

    # Apply colormap (magma), but blend with sky
    heat_rgb = cm.magma(diff_norm)[..., :3]
    alpha = diff_norm[..., None]
    blended = (1 - alpha) * sky_color + alpha * heat_rgb

    # Final panel
    panel = np.zeros((2*h, 2*w, 3), dtype=np.float32)
    panel[0:h, 0:w] = ref
    panel[0:h, w:2*w] = lower
    panel[h:2*h, 0:w] = upper
    panel[h:2*h, w:2*w] = blended

    # Save
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(panel)
    ax.axis("off")
    ax.set_title(f"Abstract Visualization idx={idx}")

    if out is None:
        out = os.path.join(abstract_dir, f"abstract_viz_{idx:06d}.png")

    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[✓] Saved: {out}")


if __name__ == "__main__":
    ### default command: python3 scripts/vis_absimg.py --config configs/${case_name}/vis_absimg.yaml
    parser = argparse.ArgumentParser(description="Visualize abstract images with YAML configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load parameters from YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    abstract_dir = config["abstract_dir"]
    index = config.get("index", None)
    out = config.get("out", None)

    print(f"index: {index}")

    if index is None:
        visualize_all(abstract_dir, out)
    else:
        visualize(abstract_dir, index, out)
