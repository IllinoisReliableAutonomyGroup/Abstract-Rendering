import argparse
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import yaml
import os,sys

grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(grandfather_path)

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from utils_save import iter_abstract_records
from DownStreamModel.gatenet.gatenet import GateNet

def default_gatenet_config(height: int, width: int) -> dict:
    """
    GateNet config matching gatenet.py training setup.
    """
    return {
        "input_shape": (3, height, width),
        "output_shape": (3,),  # GateNet predicts 3D quantity
        "l2_weight_decay": 1e-4,
        "batch_norm_decay": 0.99,
        "batch_norm_epsilon": 1e-3,
    }


def load_gatenet(ckpt_path: str, height: int, width: int, device: torch.device) -> torch.nn.Module:
    """
    Load a GateNet model from the checkpoint.
    """
    config = default_gatenet_config(height, width)
    model = GateNet(config)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)

    model.to(device)
    model.eval()
    return model


def compute_bounds_for_record(
    model: torch.nn.Module,
    record: dict,
    device: torch.device,
    height: int,
    width: int,
    abstract_dir: str,
    idx: int,
    debug: bool = False,
    eps: float = 0.01,
):
    """
    Compute bounds for a single abstract record.
    """
    if debug:
        # Load reference image
        ref_path = os.path.join(abstract_dir, f"ref_{idx:06d}.png")
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference image not found at {ref_path}")
        ref = Image.open(ref_path).convert("RGB")
        ref = np.asarray(ref).astype(np.float32) / 255.0

        # Resize reference image if necessary
        if ref.shape[:2] != (height, width):
            ref = np.array(Image.fromarray((ref * 255).astype(np.uint8)).resize((width, height))).astype(np.float32) / 255.0

        ref = torch.tensor(ref) # Convert to CHW format
        lower = ref - eps
        upper = ref + eps
    else:
        lower = record["lower"]
        upper = record["upper"]

    # Check and resize lower image if necessary
    if lower.shape[:2] != (height, width):
        lower = F.interpolate(
            lower.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)

    # Check and resize upper image if necessary
    if upper.shape[:2] != (height, width):
        upper = F.interpolate(
            upper.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)

    # Permute images to (3, height, width) for GateNet
    lower_r = lower.permute(2, 0, 1).unsqueeze(0).to(device)
    upper_r = upper.permute(2, 0, 1).unsqueeze(0).to(device)

    x_center = 0.5 * (lower_r + upper_r)

    ptb = PerturbationLpNorm(x_L=lower_r, x_U=upper_r)
    x = BoundedTensor(x_center, ptb)

    bounded_model = BoundedModule(model, x_center, device=device)

    out_lb, out_ub = bounded_model.compute_bounds(x=(x,), method="backward")

    with torch.no_grad():
        out_center = model(x_center)

    return out_lb.detach(), out_ub.detach(), out_center.detach()


def main(config_path):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    case_name = config["case_name"]
    odd_type = config["odd_type"]
    nn_type = config["nn_type"]
    height, width = config["height"], config["width"]
    debug = config["debug"]
    
    # Define paths
    abstract_dir = Path(f"Outputs/AbstractImages/{case_name}/{odd_type}")
    ckpt_path = Path(f"weights/{nn_type}/{case_name}/latest.pth")
    output_dir = Path(f"Outputs/Analysis/{case_name}/{odd_type}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{nn_type}_result.pt"

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_gatenet(ckpt_path, height, width, device)

    # Process abstract records
    out_lb_all = []
    out_ub_all = []
    out_center_all = []
    xl_all = []
    xu_all = []
    direction_all = []
    radius_all = []
    point_all = []

    for path, rec in iter_abstract_records(abstract_dir):
        idx = int(path.stem.split("_")[1])  # Extract index from filename
        out_lb, out_ub, out_center = compute_bounds_for_record(
            model, rec, device, height, width, abstract_dir, idx, debug
        )
        out_lb_all.append(out_lb.cpu())
        out_ub_all.append(out_ub.cpu())
        out_center_all.append(out_center.cpu())
        xl_all.append(rec["xl"].cpu().unsqueeze(1))  # Combine on dimension 1
        xu_all.append(rec["xu"].cpu().unsqueeze(1))  # Combine on dimension 1
        
        # Ensure direction has shape (3,) and append
        direction = rec["direction"]
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction, dtype=torch.float32)
        direction_all.append(direction.view(1, 3).cpu())  # Reshape to (1, 3)

        # Handle radius as a tensor and ensure it is 1-dimensional
        radius = rec["radius"]
        if not isinstance(radius, torch.Tensor):
            radius = torch.tensor(radius, dtype=torch.float32)
        radius_all.append(radius.view(1, -1).cpu())  # Combine on dimension 1

        # Handle point information
        point = rec["point"]
        if not isinstance(point, torch.Tensor):
            point = torch.tensor(point, dtype=torch.float32)
        point_all.append(point.unsqueeze(1).cpu())  # Combine on dimension 1

        print(f"Processed {path.name}")

    if not out_lb_all:
        print(f"No abstract_*.pt files found in {abstract_dir}")
        return

    # Combine results
    results = {
        "out_lb": torch.cat(out_lb_all, dim=0),
        "out_ub": torch.cat(out_ub_all, dim=0),
        "out_center": torch.cat(out_center_all, dim=0),
        "xl": torch.cat(xl_all, dim=1).transpose(-1, -2),  # Combine on dimension 1
        "xu": torch.cat(xu_all, dim=1).transpose(-1, -2),  # Combine on dimension 1
        "direction": torch.cat(direction_all, dim=0),  # Combine on dimension 0 to get (98, 3)
        "radius": torch.cat(radius_all, dim=0),  # Combine on dimension 1
        "point": torch.cat(point_all, dim=1).transpose(-1, -2),  # Combine on dimension 1
    }

    # Save results
    torch.save(results, output_file)
    print(f"Saved GateNet bounds and additional data to {output_file}")


if __name__ == "__main__":
    ### Default command: python3 scripts/certify_gatenet.py --config configs/${case_name}/gatenet.yml --debug
    parser = argparse.ArgumentParser(description="Certify GateNet with abstract records.")
    parser.add_argument("--config", type=str, required=True, help="Path to the gatenet.yml configuration file.")
    args = parser.parse_args()

    main(args.config)