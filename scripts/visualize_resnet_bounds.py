import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import numpy as np
import torch

# Make repo root importable (grandfather path trick)
grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
if grandfather_path not in sys.path:
    sys.path.append(grandfather_path)


# CIFAR-10 class names (for prettier x-axis labels)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_abstract_record(abstract_dir, idx):
    """Load abstract_{idx}.pt and return record dict."""
    rec_path = os.path.join(abstract_dir, f"abstract_{idx:06d}.pt")
    if not os.path.exists(rec_path):
        raise FileNotFoundError(f"Abstract record not found: {rec_path}")
    rec = torch.load(rec_path, map_location="cpu")
    return rec


def load_resnet_bounds(abstract_dir):
    """Load resnet_bounds.pt from the abstract directory."""
    bounds_path = os.path.join(abstract_dir, "resnet_bounds.pt")
    if not os.path.exists(bounds_path):
        raise FileNotFoundError(f"resnet_bounds.pt not found at {bounds_path}")
    d = torch.load(bounds_path, map_location="cpu")
    return d


def visualize_resnet_bounds(abstract_dir, idx, out_path=None):
    """
    For a given abstract sample index, plot lower/upper logits for each class,
    together with the nominal logits and predicted class.

    Uses:
      abstract_{idx}.pt  -> xl, xu
      resnet_bounds.pt   -> logits_lb[i], logits_ub[i], logits_nom[i], pred_nom[i]
    """
    rec = load_abstract_record(abstract_dir, idx)
    bounds = load_resnet_bounds(abstract_dir)

    logits_lb = bounds["logits_lb"][idx].numpy()  # (10,)
    logits_ub = bounds["logits_ub"][idx].numpy()  # (10,)
    logits_nom = bounds["logits_nom"][idx].numpy()  # (10,)
    pred_nom = int(bounds["pred_nom"][idx].item())

    # Extract xl, xu if present
    xl = rec.get("xl", None)
    xu = rec.get("xu", None)
    if xl is not None:
        xl = xl.numpy()
    if xu is not None:
        xu = xu.numpy()

    num_classes = logits_lb.shape[0]
    x = np.arange(num_classes)

    # prepare figure
    fig, ax = plt.subplots(figsize=(8, 4))

    # center & error for intervals
    centers = 0.5 * (logits_lb + logits_ub)
    half_widths = 0.5 * (logits_ub - logits_lb)

    # plot as error bars (intervals)
    ax.errorbar(
        x,
        centers,
        yerr=half_widths,
        fmt="o",
        capsize=4,
        label="logit interval [lb, ub]",
    )

    # overlay nominal logits as crosses
    ax.scatter(x, logits_nom, marker="x", color="black", label="nominal logit")

    # highlight predicted class
    ax.axvspan(pred_nom - 0.5, pred_nom + 0.5, color="green", alpha=0.1)
    pred_name = CIFAR10_CLASSES[pred_nom] if pred_nom < len(CIFAR10_CLASSES) else str(pred_nom)

    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR10_CLASSES[:num_classes], rotation=45, ha="right")
    ax.set_ylabel("logit value")
    ax.set_title(f"ResNet bounds for abstract index={idx} (pred={pred_name})")
    ax.legend(loc="best")

    # Optional: show xl/xu in text box if available
    info_lines = []
    if xl is not None and xu is not None:
        info_lines.append(f"x_l: {np.array2string(xl, precision=3)}")
        info_lines.append(f"x_u: {np.array2string(xu, precision=3)}")
    info = "\n".join(info_lines)
    if info:
        ax.text(
            0.01,
            0.99,
            info,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(abstract_dir, f"resnet_bounds_viz_{idx:06d}.png")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved ResNet bounds visualization to: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize downstream ResNet verification bounds for a single abstract sample."
    )
    parser.add_argument(
        "--abstract-dir",
        type=str,
        required=True,
        help="Directory containing abstract_*.pt and resnet_bounds.pt",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Abstract index (0 -> abstract_000000.pt).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output PNG path.",
    )

    args = parser.parse_args()
    visualize_resnet_bounds(args.abstract_dir, args.index, args.out)