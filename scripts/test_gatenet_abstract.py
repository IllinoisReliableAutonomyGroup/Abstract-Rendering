import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from matplotlib.patches import Patch

grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
if grandfather_path not in sys.path:
    sys.path.append(grandfather_path)

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from DownStreamModel.gatenet.gatenet import GateNet
from plot_gatenet import plot_filled_cuboid_fast, plot_filled_cylinder_fast, plot_gate_poses

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gatenet(config):
    model_config = {
        "input_shape": (3, config["image_height"], config["image_width"]),
        "output_shape": (3,),
        "batch_norm_decay": config["batch_norm_decay"],
        "batch_norm_epsilon": config["batch_norm_epsilon"],
    }
    model = GateNet(model_config).to(DEVICE)

    run_datetime = config.get("run_datetime")
    base_ckpt_dir = config["checkpoint_dir"]
    if run_datetime:
        ckpt_path = os.path.join(base_ckpt_dir, run_datetime, "latest.pth")
    else:
        ckpt_path = os.path.join(base_ckpt_dir, "latest.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded GateNet from: {ckpt_path}")
    return model


def crown_propagate(model, lower_img, upper_img, bound_method="backward"):
    img_center = (lower_img + upper_img) / 2.0
    bound_opts = {"conv_mode": "patches"}
    lirpa_model = BoundedModule(model, img_center, bound_opts=bound_opts, device=DEVICE)
    ptb = PerturbationLpNorm(x_L=lower_img, x_U=upper_img)
    img_ptb = BoundedTensor(img_center, ptb)
    with torch.no_grad():
        Y_lower, Y_upper = lirpa_model.compute_bounds(x=(img_ptb,), method=bound_method)
    return Y_lower, Y_upper


def main(config_path, traj_path=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=== Config ===")
    for k, v in config.items():
        print(f"  {k}: {v}")

    odd_type = config.get("odd_type", "cuboid")
    bound_method = config.get("bound_method", "backward")
    tolerance = config.get("tolerance", 0.02)
    target_size = (config["image_width"], config["image_height"])
    abstract_folder = config["abstract_folder"]
    run_datetime = config.get("run_datetime")

    # Load model
    model = load_gatenet(config)

    # Load included abstract .pt files (lower_rel/upper_rel not None)
    all_pt_files = sorted([f for f in os.listdir(abstract_folder) if f.endswith(".pt")])
    pt_files = []
    skipped = 0
    for fname in all_pt_files:
        data = torch.load(os.path.join(abstract_folder, fname), weights_only=False)
        if data.get("lower_rel") is not None and data.get("upper_rel") is not None:
            pt_files.append(fname)
        else:
            skipped += 1
    print(f"\nFound {len(pt_files)} included .pt files ({skipped} skipped — no lower_rel/upper_rel)")

    # Run CROWN on each partition cell and classify
    results = []
    n_green = 0
    n_red = 0

    for fname in tqdm(pt_files, desc="Running CROWN"):
        data = torch.load(os.path.join(abstract_folder, fname), weights_only=False)

        # Load and prepare images: [H, W, 3] → [1, 3, H, W]
        lower_img = data["lower"].permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        upper_img = data["upper"].permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # Resize if needed
        if lower_img.shape[-2:] != (target_size[1], target_size[0]):
            lower_img = F.interpolate(lower_img, size=(target_size[1], target_size[0]), mode="bilinear", align_corners=False)
            upper_img = F.interpolate(upper_img, size=(target_size[1], target_size[0]), mode="bilinear", align_corners=False)

        # CROWN propagation
        Y_lower, Y_upper = crown_propagate(model, lower_img, upper_img, bound_method)
        Y_lower = Y_lower.squeeze(0).cpu()
        Y_upper = Y_upper.squeeze(0).cpu()

        # GT bounds
        X_lower = data["lower_rel"]
        X_upper = data["upper_rel"]
        if not isinstance(X_lower, torch.Tensor):
            X_lower = torch.tensor(X_lower)
        if not isinstance(X_upper, torch.Tensor):
            X_upper = torch.tensor(X_upper)

        # Certified if predicted bounds contained within GT bounds (+/- tolerance)
        certified = bool(torch.all(Y_lower >= X_lower - tolerance) and torch.all(Y_upper <= X_upper + tolerance))
        color = "green" if certified else "red"
        if certified:
            n_green += 1
        else:
            n_red += 1

        results.append({
            "point":     data["point"],
            "direction": data["direction"],
            "radius":    data["radius"],
            "xl":        data["xl"],
            "xu":        data["xu"],
            "color":     color,
        })

    print(f"\n=== Results ===")
    print(f"Certified (green): {n_green}/{len(results)} ({100*n_green/max(len(results),1):.1f}%)")
    print(f"Violated  (red):   {n_red}/{len(results)} ({100*n_red/max(len(results),1):.1f}%)")

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for r in results:
        base      = r["point"].cpu().numpy()      if torch.is_tensor(r["point"])     else np.array(r["point"])
        direction = r["direction"].cpu().numpy()  if torch.is_tensor(r["direction"]) else np.array(r["direction"])
        radius    = r["radius"].cpu().numpy()     if torch.is_tensor(r["radius"])    else np.array(r["radius"])
        xl        = r["xl"].cpu().numpy()         if torch.is_tensor(r["xl"])        else np.array(r["xl"])
        xu        = r["xu"].cpu().numpy()         if torch.is_tensor(r["xu"])        else np.array(r["xu"])

        if odd_type == "cuboid":
            plot_filled_cuboid_fast(ax, base, direction, radius, xl, xu, color=r["color"])
        else:
            plot_filled_cylinder_fast(ax, base, direction, radius, xl, xu, color=r["color"])

    # Gate poses
    if traj_path is not None:
        with open(traj_path, encoding='utf-8') as f:
            traj_data = yaml.safe_load(f)
        gate_poses = traj_data.get("gate_poses", [])
        gate_radius = traj_data.get("gate_radius", 0.5)
        plot_gate_poses(ax, gate_poses, gate_radius=gate_radius)

    # Legend
    legend_elements = [
        Patch(facecolor="green", label=f"Certified ({n_green})"),
        Patch(facecolor="red",   label=f"Violated ({n_red})"),
    ]
    ax.legend(handles=legend_elements)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Abstract GateNet certification ({n_green}/{len(results)} certified, tol={tolerance})")

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()
    ax.set_box_aspect([1, 1, (z_limits[1] - z_limits[0]) / 3])

    # Save
    output_dir = Path(os.path.join(os.path.dirname(config["checkpoint_dir"]), "analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if run_datetime:
        output_file = output_dir / f"abstract_test_{run_datetime}_tol{tolerance}.png"
    else:
        output_file = output_dir / f"abstract_test_tol{tolerance}.png"

    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"Saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    ### python3 scripts/test_gatenet_abstract.py --config configs/${case_name}/in_certify_config.yml --traj configs/${case_name}/traj.yaml
    parser = argparse.ArgumentParser(description="Test GateNet on abstract images and visualize certification.")
    parser.add_argument("--config", type=str, required=True, help="Path to train/certify config YAML")
    parser.add_argument("--traj",   type=str, default=None,  help="Path to traj.yaml for gate poses")
    args = parser.parse_args()

    main(args.config, args.traj)