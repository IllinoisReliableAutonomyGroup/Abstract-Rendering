import os
import sys
import glob

import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib import cm

# ------------------------------------------------------------------
# Ensure repo root is on sys.path (as your advisor requested)
# ------------------------------------------------------------------
grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
if grandfather_path not in sys.path:
    sys.path.append(grandfather_path)


def _load_abstract_paths(abstract_dir):
    """Return a sorted list of abstract_*.pt paths."""
    pattern = os.path.join(abstract_dir, "abstract_*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No abstract_*.pt files found in {abstract_dir}")
    return paths


def _load_ref_image(abstract_dir, idx, fallback_shape=None):
    """
    Try to load ref_{idx}.png from abstract_dir.
    If missing, optionally return a gray placeholder.
    """
    ref_path = os.path.join(abstract_dir, f"ref_{idx}.png")
    if os.path.exists(ref_path):
        from PIL import Image
        ref = Image.open(ref_path).convert("RGB")
        ref = np.asarray(ref) / 255.0
        return ref
    if fallback_shape is not None:
        h, w, _ = fallback_shape
        return np.ones((h, w, 3), dtype=np.float32) * 0.5  # mid-gray placeholder
    return None


def _compute_global_diff_max(abstract_paths):
    """Compute a global max of |upper - lower| over all abstract records."""
    diff_max = 0.0
    for p in abstract_paths:
        rec = torch.load(p, map_location="cpu")
        lower = rec["lower"].float()  # H,W,3
        upper = rec["upper"].float()
        diff = (upper - lower).abs().mean(dim=2)  # H,W
        diff_max = max(diff_max, float(diff.max()))
    if diff_max == 0.0:
        diff_max = 1.0  # avoid division by zero
    return diff_max


def _make_frame(rec, ref_img, diff_max, cmap="magma"):
    """
    Build a single RGB frame (2x2 tile) from one abstract record:
      [ ref      | lower ]
      [ upper    | diff  ]
    """
    lower = rec["lower"].cpu().numpy()  # H,W,3
    upper = rec["upper"].cpu().numpy()

    # Clamp to [0,1] for display
    lower_img = np.clip(lower, 0.0, 1.0)
    upper_img = np.clip(upper, 0.0, 1.0)

    h, w, _ = lower_img.shape

    if ref_img is None:
        ref_img = np.ones((h, w, 3), dtype=np.float32) * 0.5
    else:
        # Ensure ref matches size
        if ref_img.shape[:2] != (h, w):
            from PIL import Image
            ref_pil = Image.fromarray((ref_img * 255).astype(np.uint8))
            ref_pil = ref_pil.resize((w, h), Image.BILINEAR)
            ref_img = np.asarray(ref_pil) / 255.0

    # Uncertainty heatmap
    diff = np.abs(upper_img - lower_img).mean(axis=2)  # H,W
    diff_norm = np.clip(diff / diff_max, 0.0, 1.0)

    # Apply colormap to get RGB
    colormap = cm.get_cmap(cmap)
    diff_rgb = colormap(diff_norm)[..., :3]  # drop alpha

    # Build 2x2 tile
    frame = np.zeros((2 * h, 2 * w, 3), dtype=np.float32)

    # Top row
    frame[0:h, 0:w, :] = ref_img         # top-left
    frame[0:h, w:2*w, :] = lower_img     # top-right

    # Bottom row
    frame[h:2*h, 0:w, :] = upper_img     # bottom-left
    frame[h:2*h, w:2*w, :] = diff_rgb    # bottom-right

    frame = np.clip(frame, 0.0, 1.0)
    return (frame * 255).astype(np.uint8)


def animate_uncertainty_vs_yaw(
    abstract_dir: str,
    out_name: str = "uncertainty_vs_yaw.gif",
    fps: int = 15,
    max_frames: int | None = None,
):
    """
    Create a GIF that visualizes how uncertainty (|upper - lower|) evolves
    across abstract records (interpreted as sampling along yaw).

    The output frames are high-resolution 2x2 tiles:
        [ ref | lower ]
        [ upper | diff heatmap ]
    """
    abstract_paths = _load_abstract_paths(abstract_dir)

    if max_frames is not None:
        abstract_paths = abstract_paths[:max_frames]

    print(f"Found {len(abstract_paths)} abstract records.")

    # Global max for consistent colormap scaling
    print("Computing global max |upper - lower|...")
    diff_max = _compute_global_diff_max(abstract_paths)
    print(f"Global diff_max = {diff_max:.6f}")

    frames = []

    for i, p in enumerate(abstract_paths):
        rec = torch.load(p, map_location="cpu")
        # index from filename, assuming abstract_000123.pt
        basename = os.path.basename(p)
        idx_str = basename.replace("abstract_", "").replace(".pt", "")
        try:
            idx = int(idx_str)
        except ValueError:
            idx = i

        ref_img = _load_ref_image(abstract_dir, idx, fallback_shape=rec["lower"].shape)
        frame = _make_frame(rec, ref_img, diff_max, cmap="magma")
        frames.append(frame)

        print(f"[{i+1}/{len(abstract_paths)}] added frame for {basename}")

    out_path = os.path.join(abstract_dir, out_name)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"\nSaved GIF to: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--abstract-dir",
        type=str,
        required=True,
        help="Directory containing abstract_*.pt and optional ref_*.png.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for the GIF.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="If set, only use the first N abstract_*.pt files.",
    )
    args = parser.parse_args()

    animate_uncertainty_vs_yaw(
        abstract_dir=args.abstract_dir,
        out_name="uncertainty_vs_yaw.gif",
        fps=args.fps,
        max_frames=args.max_frames,
    )