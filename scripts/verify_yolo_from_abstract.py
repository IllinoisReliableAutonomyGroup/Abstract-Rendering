import argparse
import os
import sys
from pathlib import Path

grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
if grandfather_path not in sys.path:
    sys.path.append(grandfather_path)

from verify_nn_from_abstract import verify_yolo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--abstract-dir", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=416)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    verify_yolo(args)


class TinyYOLOWrapper(nn.Module):
    """
    Wrap the TinyYOLO ONNX model so that:
      - input:  (B, 3, H, W), float in [0,1]
      - output: (B, N) flattened vector (good for LiRPA)
    """

    def __init__(self, onnx_path: str):
        super().__init__()
        onnx_model = onnx.load(onnx_path)
        self.model = ConvertModel(onnx_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        y = self.model(x)
        # y might be (B, C, H', W') or already (B, N)
        if y.dim() > 2:
            y = y.view(y.size(0), -1)
        return y


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
    bounded_model: BoundedModule,
    base_model: nn.Module,
    record: dict,
    device: torch.device,
    img_size: int,
):
    """
    record: abstract record with 'lower'/'upper' in (H,W,3) [0,1].

    Returns:
      out_lb:  (1, N) lower bounds on flattened YOLO output
      out_ub:  (1, N) upper bounds
      out_nom: (1, N) nominal output at center of interval
    """
    lower = record["lower"]
    upper = record["upper"]

    # Resize bounds to YOLO input size
    lower_r = resize_to_square(lower, img_size).to(device)
    upper_r = resize_to_square(upper, img_size).to(device)

    # Center and element-wise box perturbation
    x_center = 0.5 * (lower_r + upper_r)
    ptb = PerturbationLpNorm(x_L=lower_r, x_U=upper_r)
    x = BoundedTensor(x_center, ptb)

    out_lb, out_ub = bounded_model.compute_bounds(x=(x,), method="backward")

    with torch.no_grad():
        out_nom = base_model(x_center)

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
        "--onnx",
        type=str,
        required=True,
        help="Path to TinyYOLO.onnx "
             "(e.g., DownStreamModel/yolo/TinyYOLO.onnx)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=416,
        help="YOLO input resolution (e.g., 416). Adjust if your TinyYOLO uses a different size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu).",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, only process the first N abstract_*.pt records.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build YOLO model and bounded wrapper
    yolo = TinyYOLOWrapper(args.onnx).to(device)
    yolo.eval()

    dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
    bounded_yolo = BoundedModule(yolo, dummy, device=device)

    abstract_dir = Path(args.abstract_dir)
    out_lb_all = []
    out_ub_all = []
    out_nom_all = []

    for idx, (path, rec) in enumerate(iter_abstract_records(abstract_dir)):
        out_lb, out_ub, out_nom = compute_bounds_for_record(
            bounded_yolo, yolo, rec, device, args.img_size
        )
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

    out_path = abstract_dir / "yolo_bounds.pt"
    torch.save(
        {
            "out_lb": out_lb_all,
            "out_ub": out_ub_all,
            "out_nom": out_nom_all,
        },
        out_path,
    )
    print(f"Saved YOLO bounds to {out_path}")


if __name__ == "__main__":
    main()