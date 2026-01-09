import argparse
from pathlib import Path
import os
import sys

# Ensure repo root is importable
grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
if grandfather_path not in sys.path:
    sys.path.append(grandfather_path)

import torch
import torch.nn.functional as F
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from utils import iter_abstract_records

# ResNet imports
from DownStreamModel.cifar10_resnet.resnet import resnet2b, resnet4b

# GateNet imports
from DownStreamModel.gatenet.gatenet import GateNet

# YOLO imports
import onnx
from onnx2pytorch import ConvertModel

# Compatibility patch for some onnx2pytorch versions
import onnx2pytorch.operations.pad as pad_module
import onnx2pytorch.convert.operations as conv_ops
_OrigPad = pad_module.Pad
class CompatPad(_OrigPad):
    def __init__(self, *args, constant=None, value=None, **kwargs):
        super().__init__(*args, **kwargs)
pad_module.Pad = CompatPad
conv_ops.Pad = CompatPad


# CIFAR-10 statistics used by the ResNet models
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)


def ensure_output_dir(output_root: Path, abstract_dir: Path) -> Path:
    """Create and return a model-specific output directory under Outputs/NNCertification.

    Example: Outputs/NNCertification/airplane_grey_round
    """
    repo_root = Path(grandfather_path)
    out_base = repo_root / output_root
    # Choose a readable subfolder name combining parent and leaf of abstract_dir
    parent = abstract_dir.parent.name
    leaf = abstract_dir.name
    sub = f"{parent}_{leaf}"
    out_dir = out_base / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ----- ResNet helpers -----

def load_resnet(model_name: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    if model_name == "resnet2b":
        model = resnet2b()
    elif model_name == "resnet4b":
        model = resnet4b()
    else:
        raise ValueError(f"Unknown model name {model_name}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    return model


def normalize_bounds(lower: torch.Tensor, upper: torch.Tensor, device: torch.device):
    mean = CIFAR_MEAN.to(device)
    std = CIFAR_STD.to(device)
    lower_n = (lower - mean) / std
    upper_n = (upper - mean) / std
    lb = torch.min(lower_n, upper_n)
    ub = torch.max(lower_n, upper_n)
    return lb, ub


def compute_resnet_bounds(model: torch.nn.Module, record: dict, device: torch.device):
    lower = record["lower"].permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    upper = record["upper"].permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    target_size = (32, 32)
    lower = F.interpolate(lower, size=target_size, mode="bilinear", align_corners=False)
    upper = F.interpolate(upper, size=target_size, mode="bilinear", align_corners=False)
    lower_n, upper_n = normalize_bounds(lower, upper, device)
    x_center = 0.5 * (lower_n + upper_n)
    ptb = PerturbationLpNorm(x_L=lower_n, x_U=upper_n)
    x = BoundedTensor(x_center, ptb)
    bounded_model = BoundedModule(model, x_center, device=device)
    logits_lb, logits_ub = bounded_model.compute_bounds(x=(x,), method="backward")
    with torch.no_grad():
        logits_nom = model(x_center)
        pred_nom = logits_nom.argmax(dim=1)
    return logits_lb.detach(), logits_ub.detach(), logits_nom.detach(), pred_nom.detach()


# ----- GateNet helpers -----

def default_gatenet_config(img_size: int) -> dict:
    return {
        "input_shape": (3, img_size, img_size),
        "output_shape": (3,),
        "l2_weight_decay": 1e-4,
        "batch_norm_decay": 0.99,
        "batch_norm_epsilon": 1e-3,
    }


def load_gatenet(ckpt_path: str, img_size: int, device: torch.device) -> torch.nn.Module:
    config = default_gatenet_config(img_size)
    model = GateNet(config)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "net"]:
            if key in ckpt:
                state_dict = ckpt[key]
                break
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    return model


def resize_to_square(img_hwc: torch.Tensor, size: int) -> torch.Tensor:
    img_chw = img_hwc.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    img_resized = F.interpolate(img_chw, size=(size, size), mode="bilinear", align_corners=False)
    return img_resized


def compute_gatenet_bounds(model: torch.nn.Module, record: dict, device: torch.device, img_size: int):
    lower = record["lower"]
    upper = record["upper"]
    lower_r = resize_to_square(lower, img_size).to(device)
    upper_r = resize_to_square(upper, img_size).to(device)
    x_center = 0.5 * (lower_r + upper_r)
    ptb = PerturbationLpNorm(x_L=lower_r, x_U=upper_r)
    x = BoundedTensor(x_center, ptb)
    bounded_model = BoundedModule(model, x_center, device=device)
    out_lb, out_ub = bounded_model.compute_bounds(x=(x,), method="backward")
    with torch.no_grad():
        out_nom = model(x_center)
    return out_lb.detach(), out_ub.detach(), out_nom.detach()


# ----- YOLO helpers -----

class TinyYOLOWrapper(nn.Module):
    def __init__(self, onnx_path: str):
        super().__init__()
        onnx_model = onnx.load(onnx_path)
        self.model = ConvertModel(onnx_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        if y.dim() > 2:
            y = y.view(y.size(0), -1)
        return y


def compute_yolo_bounds(bounded_model: BoundedModule, base_model: nn.Module, record: dict, device: torch.device, img_size: int):
    lower = record["lower"]
    upper = record["upper"]
    lower_r = resize_to_square(lower, img_size).to(device)
    upper_r = resize_to_square(upper, img_size).to(device)
    x_center = 0.5 * (lower_r + upper_r)
    ptb = PerturbationLpNorm(x_L=lower_r, x_U=upper_r)
    x = BoundedTensor(x_center, ptb)
    out_lb, out_ub = bounded_model.compute_bounds(x=(x,), method="backward")
    with torch.no_grad():
        out_nom = base_model(x_center)
    return out_lb.detach(), out_ub.detach(), out_nom.detach()


# ----- Top-level verification entry points (callable from wrappers) -----

def verify_resnet(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_resnet(args.model_name, args.ckpt, device)
    abstract_dir = Path(args.abstract_dir)

    logits_lb_all = []
    logits_ub_all = []
    logits_nom_all = []
    pred_nom_all = []

    for path, rec in iter_abstract_records(abstract_dir):
        logits_lb, logits_ub, logits_nom, pred_nom = compute_resnet_bounds(model, rec, device)
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

    out_dir = ensure_output_dir(Path("Outputs") / "NNCertification", abstract_dir)
    out_path = out_dir / f"{abstract_dir.parent.name}_{abstract_dir.name}_resnet_bounds.pt"

    torch.save({
        "logits_lb": logits_lb_all,
        "logits_ub": logits_ub_all,
        "logits_nom": logits_nom_all,
        "pred_nom": pred_nom_all,
    }, out_path)
    print(f"Saved ResNet classification bounds to {out_path}")


def verify_gatenet(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_gatenet(args.ckpt, args.img_size, device)
    abstract_dir = Path(args.abstract_dir)

    out_lb_all = []
    out_ub_all = []
    out_nom_all = []

    for path, rec in iter_abstract_records(abstract_dir):
        out_lb, out_ub, out_nom = compute_gatenet_bounds(model, rec, device, args.img_size)
        out_lb_all.append(out_lb.cpu())
        out_ub_all.append(out_ub.cpu())
        out_nom_all.append(out_nom.cpu())
        print(f"Processed {path.name}")

    if not out_lb_all:
        print(f"No abstract_*.pt files found in {abstract_dir}")
        return

    out_lb_all = torch.cat(out_lb_all, dim=0)
    out_ub_all = torch.cat(out_ub_all, dim=0)
    out_nom_all = torch.cat(out_nom_all, dim=0)

    out_dir = ensure_output_dir(Path("Outputs") / "NNCertification", abstract_dir)
    out_path = out_dir / f"{abstract_dir.parent.name}_{abstract_dir.name}_gatenet_bounds.pt"

    torch.save({
        "out_lb": out_lb_all,
        "out_ub": out_ub_all,
        "out_nom": out_nom_all,
    }, out_path)
    print(f"Saved GateNet bounds to {out_path}")


def verify_yolo(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    yolo = TinyYOLOWrapper(args.onnx).to(device)
    yolo.eval()
    dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
    bounded_yolo = BoundedModule(yolo, dummy, device=device)

    abstract_dir = Path(args.abstract_dir)
    out_lb_all = []
    out_ub_all = []
    out_nom_all = []

    for idx, (path, rec) in enumerate(iter_abstract_records(abstract_dir)):
        out_lb, out_ub, out_nom = compute_yolo_bounds(bounded_yolo, yolo, rec, device, args.img_size)
        out_lb_all.append(out_lb.cpu())
        out_ub_all.append(out_ub.cpu())
        out_nom_all.append(out_nom.cpu())
        print(f"Processed {path.name}")
        if args.max_samples is not None and (idx + 1) >= args.max_samples:
            break

    if not out_lb_all:
        print(f"No abstract_*.pt files found in {abstract_dir}")
        return

    out_lb_all = torch.cat(out_lb_all, dim=0)
    out_ub_all = torch.cat(out_ub_all, dim=0)
    out_nom_all = torch.cat(out_nom_all, dim=0)

    out_dir = ensure_output_dir(Path("Outputs") / "NNCertification", abstract_dir)
    out_path = out_dir / f"{abstract_dir.parent.name}_{abstract_dir.name}_yolo_bounds.pt"

    torch.save({
        "out_lb": out_lb_all,
        "out_ub": out_ub_all,
        "out_nom": out_nom_all,
    }, out_path)
    print(f"Saved YOLO bounds to {out_path}")


# ----- CLI -----

def main():
    parser = argparse.ArgumentParser("Unified verification for ResNet, GateNet, and YOLO using abstract records")
    sub = parser.add_subparsers(dest="which", required=True)

    # ResNet
    p_res = sub.add_parser("resnet")
    p_res.add_argument("--abstract-dir", type=str, required=True)
    p_res.add_argument("--model-name", type=str, default="resnet4b", choices=["resnet2b", "resnet4b"])
    p_res.add_argument("--ckpt", type=str, required=True)
    p_res.add_argument("--device", type=str, default="cuda")
    p_res.set_defaults(func=verify_resnet)

    # GateNet
    p_gate = sub.add_parser("gatenet")
    p_gate.add_argument("--abstract-dir", type=str, required=True)
    p_gate.add_argument("--ckpt", type=str, required=True)
    p_gate.add_argument("--img-size", type=int, default=64)
    p_gate.add_argument("--device", type=str, default="cuda")
    p_gate.set_defaults(func=verify_gatenet)

    # YOLO
    p_yolo = sub.add_parser("yolo")
    p_yolo.add_argument("--abstract-dir", type=str, required=True)
    p_yolo.add_argument("--onnx", type=str, required=True)
    p_yolo.add_argument("--img-size", type=int, default=416)
    p_yolo.add_argument("--device", type=str, default="cuda")
    p_yolo.add_argument("--max-samples", type=int, default=None)
    p_yolo.set_defaults(func=verify_yolo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
