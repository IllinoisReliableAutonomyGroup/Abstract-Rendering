import argparse
from pathlib import Path

import os,sys

grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(grandfather_path)

import torch
import torch.nn.functional as F

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from utils import iter_abstract_records
from DownStreamModel.cifar10_resnet.resnet import resnet2b, resnet4b


# CIFAR-10 statistics used by the ResNet models
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)


def load_resnet(model_name: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load a ResNet model (resnet2b or resnet4b) from DownStreamModel/cifar10_resnet.
    """
    if model_name == "resnet2b":
        model = resnet2b()
    elif model_name == "resnet4b":
        model = resnet4b()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # checkpoint may be a plain state_dict or wrapped in a dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # strip optional "module." prefix from DataParallel
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)

    model.to(device)
    model.eval()
    return model


def normalize_bounds(lower: torch.Tensor, upper: torch.Tensor, device: torch.device):
    """
    lower, upper: tensors in [0,1], shape (1,3,H,W).
    Apply CIFAR normalization interval-wise and ensure lower <= upper.
    """
    mean = CIFAR_MEAN.to(device)
    std = CIFAR_STD.to(device)

    lower_n = (lower - mean) / std
    upper_n = (upper - mean) / std

    lb = torch.min(lower_n, upper_n)
    ub = torch.max(lower_n, upper_n)
    return lb, ub


def compute_bounds_for_record(
    model: torch.nn.Module,
    record: dict,
    device: torch.device,
):
    """
    record: abstract record with 'lower'/'upper' in (H,W,3) [0,1].

    Returns:
      logits_lb: (1, num_classes) lower bounds on logits
      logits_ub: (1, num_classes) upper bounds on logits
      logits_nom: (1, num_classes) nominal logits at center of interval
      pred_nom: (1,) nominal predicted class index
    """
    # (H, W, 3) -> (1, 3, H, W)
    lower = record["lower"].permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    upper = record["upper"].permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)

    # 🔹 NEW: resize to CIFAR-10 resolution expected by resnet2b/4b
    target_size = (32, 32)
    lower = F.interpolate(lower, size=target_size, mode="bilinear", align_corners=False)
    upper = F.interpolate(upper, size=target_size, mode="bilinear", align_corners=False)

    # then normalize
    lower_n, upper_n = normalize_bounds(lower, upper, device)

    # center of the interval
    x_center = 0.5 * (lower_n + upper_n)

    # element-wise box perturbation
    ptb = PerturbationLpNorm(x_L=lower_n, x_U=upper_n)
    x = BoundedTensor(x_center, ptb)

    # build a BoundedModule for this input shape
    bounded_model = BoundedModule(model, x_center, device=device)

    # compute bounds on logits
    logits_lb, logits_ub = bounded_model.compute_bounds(x=(x,), method="backward")

    # nominal logits & prediction at the center
    with torch.no_grad():
        logits_nom = model(x_center)
        pred_nom = logits_nom.argmax(dim=1)

    return (
        logits_lb.detach(),
        logits_ub.detach(),
        logits_nom.detach(),
        pred_nom.detach(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--abstract-dir",
        type=str,
        required=True,
        help="Folder with abstract_*.pt from abstract_gsplat.py "
             "(e.g., Outputs/AbstractImages/airplane_grey)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet4b",
        choices=["resnet2b", "resnet4b"],
        help="Which ResNet to use.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to ResNet checkpoint (e.g., DownStreamModel/cifar10_resnet/resnet4b.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu).",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_resnet(args.model_name, args.ckpt, device)

    abstract_dir = Path(args.abstract_dir)

    logits_lb_all = []
    logits_ub_all = []
    logits_nom_all = []
    pred_nom_all = []

    for path, rec in iter_abstract_records(abstract_dir):
        logits_lb, logits_ub, logits_nom, pred_nom = compute_bounds_for_record(
            model, rec, device
        )
        logits_lb_all.append(logits_lb.cpu())
        logits_ub_all.append(logits_ub.cpu())
        logits_nom_all.append(logits_nom.cpu())
        pred_nom_all.append(pred_nom.cpu())

        print(f"{path.name}: nominal pred = {int(pred_nom.item())}")

    if not logits_lb_all:
        print(f"No abstract_*.pt files found in {abstract_dir}")
        return

    logits_lb_all = torch.cat(logits_lb_all, dim=0)
    logits_ub_all = torch.cat(logits_ub_all, dim=0)
    logits_nom_all = torch.cat(logits_nom_all, dim=0)
    pred_nom_all = torch.cat(pred_nom_all, dim=0)

    out_path = abstract_dir / "resnet_bounds.pt"
    torch.save(
        {
            "logits_lb": logits_lb_all,
            "logits_ub": logits_ub_all,
            "logits_nom": logits_nom_all,
            "pred_nom": pred_nom_all,
        },
        out_path,
    )
    print(f"Saved classification bounds to {out_path}")


if __name__ == "__main__":
    main()