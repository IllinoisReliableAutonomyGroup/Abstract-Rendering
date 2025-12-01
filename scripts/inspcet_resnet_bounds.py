import torch
from pathlib import Path

def load_resnet_bounds(bounds_path: str):
    """
    Load the saved classification bound file produced by the main script.

    Expected structure of resnet_bounds.pt:
        {
            "logits_lb":  (N, num_classes),
            "logits_ub":  (N, num_classes),
            "logits_nom": (N, num_classes),
            "pred_nom":   (N,)
        }

    Returns: dict of tensors on CPU.
    """
    bounds_path = Path(bounds_path)
    if not bounds_path.exists():
        raise FileNotFoundError(f"File not found: {bounds_path}")

    data = torch.load(bounds_path, map_location="cpu")

    required_keys = ["logits_lb", "logits_ub", "logits_nom", "pred_nom"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {bounds_path}")

    print("Loaded:")
    print(f"  logits_lb : {tuple(data['logits_lb'].shape)}")
    print(f"  logits_ub : {tuple(data['logits_ub'].shape)}")
    print(f"  logits_nom: {tuple(data['logits_nom'].shape)}")
    print(f"  pred_nom  : {tuple(data['pred_nom'].shape)}")

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to resnet_bounds.pt saved by the bound computation script."
            "(e.g., Outputs/AbstractImages/airplane_grey/round/resnet_bounds.pt)",
    )
    args = parser.parse_args()

    data = load_resnet_bounds(args.path)

    # example prints
    print()
    print("Example:")
    print("Nominal prediction at index 0:", int(data["pred_nom"][0]))
    print("Logit interval at index 0:")
    print("  LB:", data["logits_lb"][0])
    print("  UB:", data["logits_ub"][0])
