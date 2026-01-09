import argparse
import os
import sys
from pathlib import Path

grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
if grandfather_path not in sys.path:
    sys.path.append(grandfather_path)

from verify_nn_from_abstract import verify_gatenet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--abstract-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    verify_gatenet(args)


def default_gatenet_config(img_size: int) -> dict:
    """
    GateNet config matching gatenet.py training setup.

    img_size: height/width of input images (e.g., 64)
    """
    return {
        "input_shape": (3, img_size, img_size),
        "output_shape": (3,),      # GateNet predicts 3D quantity in the training script
        "l2_weight_decay": 1e-4,
        "batch_norm_decay": 0.99,
        "batch_norm_epsilon": 1e-3,
    }


def load_gatenet(ckpt_path: str, img_size: int, device: torch.device) -> torch.nn.Module:
    """
    Load a GateNet model from DownStreamModel/gatenet/checkpoint_*.pth.
    """
    config = default_gatenet_config(img_size)
    model = GateNet(config)

    ckpt = torch.load(ckpt_path, map_location=device)

    # Try common checkpoint layouts
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
    """
    img_hwc: (H, W, 3), [0,1]
    returns: (1, 3, size, size)
    """
    img_chw = img_hwc.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    img_resized = F.interpolate(
        img_chw,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return img_resized


def compute_bounds_for_record(
    model: torch.nn.Module,
    record: dict,
    device: torch.device,
    img_size: int,
):
    """
    record: abstract record with 'lower'/'upper' in (H,W,3) [0,1].

    Returns:
      out_lb:  (1, D) lower bounds on regression outputs
      out_ub:  (1, D) upper bounds
      out_nom: (1, D) nominal regression output at center of interval
    """
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
        "--ckpt",
        type=str,
        required=True,
        help="Path to GateNet checkpoint "
             "(e.g., DownStreamModel/gatenet/checkpoint_epoch_100.pth)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=64,
        help="Input resolution GateNet expects (e.g., 64). "
             "Adjust if you trained with a different size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu).",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_gatenet(args.ckpt, args.img_size, device)

    abstract_dir = Path(args.abstract_dir)
    out_lb_all = []
    out_ub_all = []
    out_nom_all = []

    for path, rec in iter_abstract_records(abstract_dir):
        out_lb, out_ub, out_nom = compute_bounds_for_record(
            model, rec, device, args.img_size
        )
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

    out_path = abstract_dir / "gatenet_bounds.pt"
    torch.save(
        {
            "out_lb": out_lb_all,
            "out_ub": out_ub_all,
            "out_nom": out_nom_all,
        },
        out_path,
    )
    print(f"Saved GateNet bounds to {out_path}")


if __name__ == "__main__":
    main()