import os
import json
import sys
import yaml

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from operator import itemgetter
from collections import deque

from scipy.spatial.transform import Rotation
from PIL import Image

grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(grandfather_path)

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from collections import defaultdict

from scripts.utils_partition import generate_partition, refine_partition
from scripts.utils_alpha_blending import alpha_blending_ref, alpha_blending_ptb
from scripts.utils_save import save_abstract_record
from scripts.utils_transform import orthogonal_basis_from_direction
from render_models import GsplatRGB, TransferModel

import argparse

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32


def compute_partition_cuboid_bounds(
    base_trans, direction, rot, radius, gate_pos, input_lb, input_ub, pert_type="cylinder"
):
    """
    Compute lower/upper bounds on relative pose (gate in camera frame) for a
    partition cell.  Supports both cylinder and cuboid perturbation types.

    For cylinder:
      cam = base_trans + d*direction + r*radius*(cos(θ)*o1 + sin(θ)*o2)
      d in [lb[0], ub[0]], r in [lb[1], ub[1]], θ in [lb[2], ub[2]]
      Overapproximates circular cross-section as a square.

    For cuboid (axis-aligned, exact AABB):
      cam = base_trans + d*direction + x_norm*x_half*o1 + z_norm*z_half*o2
      d in [lb[0], ub[0]], x_norm in [lb[1], ub[1]], z_norm in [lb[2], ub[2]]
      radius is [x_half, z_half].
      Because the cuboid is axis-aligned, the AABB is exact.

    Returns None, None if gate_pos is None.
    """
    if gate_pos is None:
        return None, None

    gate_pos_t = torch.tensor(gate_pos[:3], device=base_trans.device, dtype=base_trans.dtype)

    _, o1, o2 = orthogonal_basis_from_direction(direction)

    C  = rot @ (gate_pos_t - base_trans)
    A  = rot @ direction
    B1 = rot @ o1
    B2 = rot @ o2

    dl = input_lb[0]
    du = input_ub[0]

    if pert_type == "cylinder":
        r_max = input_ub[1]
        effective_r = r_max * radius
        lower = C - torch.maximum(dl * A, du * A) - effective_r * (torch.abs(B1) + torch.abs(B2))
        upper = C - torch.minimum(dl * A, du * A) + effective_r * (torch.abs(B1) + torch.abs(B2))

    elif pert_type == "cuboid":
        # radius is [x_half, z_half]
        x_half = radius[0]
        z_half = radius[1]
        xl, xu = input_lb[1], input_ub[1]   # x_norm sub-cell bounds
        zl, zu = input_lb[2], input_ub[2]   # z_norm sub-cell bounds

        # Exact AABB: each dimension is independent so extremes are at corners
        lower = (
            C
            - torch.maximum(dl * A, du * A)
            - x_half * torch.maximum(xl * B1, xu * B1)
            - z_half * torch.maximum(zl * B2, zu * B2)
        )
        upper = (
            C
            - torch.minimum(dl * A, du * A)
            - x_half * torch.minimum(xl * B1, xu * B1)
            - z_half * torch.minimum(zl * B2, zu * B2)
        )

    else:
        raise ValueError(f"Unknown pert_type: {pert_type}")

    return lower.detach().cpu(), upper.detach().cpu()


def _save_abstract_record_with_bounds(
    save_dir, index,
    lower_input, upper_input, lower_img, upper_img,
    img_lA=None, img_uA=None, img_lbias=None, img_ubias=None,
    point=None, direction=None, radius=None,
    lower_rel=None, upper_rel=None,
):
    """
    Save abstract record with optional cuboid relative-pose bounds and linear set fields.
    """
    def _to_float_tensor(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32, copy=False))
        return x.to(dtype=torch.float32).detach().cpu()

    record = {
        "xl":        _to_float_tensor(lower_input),
        "xu":        _to_float_tensor(upper_input),
        "lower":     _to_float_tensor(lower_img),
        "upper":     _to_float_tensor(upper_img),
        "lA":        _to_float_tensor(img_lA),
        "uA":        _to_float_tensor(img_uA),
        "lb":        _to_float_tensor(img_lbias),
        "ub":        _to_float_tensor(img_ubias),
        "point":     _to_float_tensor(point),
        "direction": _to_float_tensor(direction),
        "radius":    _to_float_tensor(radius),
        "lower_rel": lower_rel,
        "upper_rel": upper_rel,
    }

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"abstract_{index:06d}.pt")
    torch.save(record, out_path)
    return out_path


def main(setup_dict):
    key_list = [
        "bound_method", "render_method", "case_name", "odd_type", "pert_type", "debug", "width", "height",
        "fx", "fy", "eps2d", "tile_size", "min_distance", "max_distance", "gs_batch",
        "part", "scene_path", "checkpoint_filename",
        "save_folder", "bg_img_path", "bg_pure_color", "save_ref", "save_bound",
        "N_samples", "poses", "radiuss", "gates", "includes"
    ]

    bound_method, render_method, case_name, odd_type, pert_type, debug, width, height, fx, fy, eps2d, \
    tile_size, min_distance, max_distance, gs_batch, part, \
    scene_path, checkpoint_filename, save_folder, bg_img_path, \
    bg_pure_color, save_ref, save_bound, N_samples, poses, radiuss, gates, includes = itemgetter(*key_list)(setup_dict)

    # Load Already Trained Scene Files
    script_dir = os.path.dirname(os.path.realpath(__file__))
    scene_folder = os.path.join(script_dir, scene_path)
    transform_file = os.path.join(scene_folder, 'dataparser_transforms.json')
    checkpoint_file = os.path.join(scene_folder, 'nerfstudio_models/', checkpoint_filename)

    # Load Transformation Matrix and Scale
    with open (transform_file, 'r') as fp:
        data_transform = json.load(fp)
        transform = np.array (data_transform['transform'])
        transform_hom = np.vstack((transform, np.array([0,0,0,1])))
        scale = data_transform['scale']

    transform_hom = torch.from_numpy(transform_hom).to(dtype=DTYPE, device=DEVICE)
    scale = torch.tensor(scale,dtype=DTYPE, device=DEVICE)

    # Make Folder to Save Abstract Images
    save_folder_full = os.path.join(script_dir, save_folder)
    if not os.path.exists(save_folder_full):
        os.makedirs(save_folder_full)

    # Load Trained 3DGS
    scene_parameters = torch.load(checkpoint_file, weights_only=False, map_location=DEVICE)
    means = scene_parameters['pipeline']['_model.gauss_params.means'].to(DEVICE)
    quats = scene_parameters['pipeline']['_model.gauss_params.quats'].to(DEVICE)
    opacities = torch.sigmoid(scene_parameters['pipeline']['_model.gauss_params.opacities']).to(DEVICE)
    scales = torch.exp(scene_parameters['pipeline']['_model.gauss_params.scales']).to(DEVICE)
    colors = torch.sigmoid(scene_parameters['pipeline']['_model.gauss_params.features_dc']).to(DEVICE)
    gauss_num = means.size(0)
    print(f"Number of Total Gaussians in the Scene: {gauss_num}")

    assert torch.all((opacities>=0) & (opacities<=1))

    # Define camera_dict
    camera_dict = {
        "fx": fx,
        "fy": fy,
        "width": width,
        "height": height,
    }

    # Define scene_dict
    scene_dict_all = {
        "means": means,
        "quats": quats,
        "opacities": opacities,
        "scales": scales,
        "colors": colors
    }

    # Define Background Image
    if bg_img_path is None:
        bg_pure_color = torch.tensor(bg_pure_color)
        bg_color = bg_pure_color.view(1, 1, 3).repeat(height, width,  1).to(DEVICE)
    else:
        bg_img = Image.open(bg_img_path).convert("RGB")  # ensure 3 channels
        bg_img = bg_img.resize((height, width), Image.LANCZOS)
        bg_img = np.array(bg_img, dtype=np.float32)  # shape: (H, W, 3)
        bg_color = torch.from_numpy(bg_img/255).to(DEVICE)  # shape: (H, W, 3)

    # Generate Rotation Matrix
    transs, orientations = np.array(poses)[:, :3], np.array(poses)[:, 3:]
    Rs = Rotation.from_euler('ZYX', orientations)
    radiuss = np.array(radiuss)
    rots = Rs.as_matrix()
    gates = list(gates)

    # Create Queue for Waypoints (now includes gate_pos and include flag)
    queue = deque(zip(transs, rots, radiuss, gates, includes))
    absimg_num = 0
    pbar = tqdm(total=len(queue)-1,desc="Processing Poses", unit="item")

    # Define Render and Verification Network
    pipeline_type = "abstract-rendering"
    render_net = GsplatRGB(camera_dict, scene_dict_all, min_distance, max_distance, bg_color, eps2d, gs_batch).to(DEVICE)
    verf_net = TransferModel(render_net, pipeline_type, None, None, transform_hom, scale, odd_type).to(DEVICE)

    # Create Partition Inputs
    inputs_center, inputs_lb, inputs_ub = generate_partition(odd_type, part) # [part, N]
    inputs_center, inputs_lb, inputs_ub= torch.tensor(inputs_center).to(device=DEVICE, dtype=DTYPE), torch.tensor(inputs_lb).to(device=DEVICE, dtype=DTYPE), torch.tensor(inputs_ub).to(device=DEVICE, dtype=DTYPE)

    while len(queue)>1:
        base_trans, rot, radius, gate_pos, include = queue.popleft()
        base_trans_next = queue[0][0]

        base_trans = torch.from_numpy(base_trans).to(device=DEVICE, dtype=DTYPE)
        base_trans_next = torch.from_numpy(base_trans_next).to(device=DEVICE, dtype=DTYPE)
        direction = base_trans_next - base_trans
        rot = torch.from_numpy(rot).to(device=DEVICE, dtype=DTYPE)
        # For cylinder: radius is scalar. For cuboid: radius is [x_half, z_half].
        radius = torch.tensor(radius, device=DEVICE, dtype=DTYPE)
        verf_net.update_model_param(rot, base_trans, direction, radius)

        # Create Queue for Inputs (Pose Cells)
        inputs_queue = deque(zip(inputs_center, inputs_lb, inputs_ub))
        pbar2 = tqdm(total=len(inputs_queue),desc="Processing Cells", unit="item")

        while inputs_queue:
            input_center_org, input_lb_org, input_ub_org = inputs_queue.popleft()
            input_center, input_lb, input_ub = refine_partition(input_center_org.clone(), input_lb_org.clone(), input_ub_org.clone(), odd_type)
            input_center, input_lb, input_ub = input_center.unsqueeze(0), input_lb.unsqueeze(0), input_ub.unsqueeze(0)

            # Compute relative-pose bounds for this partition cell (None if excluded)
            if include:
                lower_rel, upper_rel = compute_partition_cuboid_bounds(
                    base_trans, direction, rot, radius, gate_pos,
                    input_lb.squeeze(0), input_ub.squeeze(0),
                    pert_type=pert_type,
                )
            else:
                lower_rel, upper_rel = None, None

            # Sort Gaussians based on Distance to Camera
            verf_net.call_model_preprocess("sort_gauss", input_center)

            if save_ref:
                img_ref = np.zeros((height, width,3))
            if save_bound:
                img_lb = np.zeros((height, width,3))
                img_ub = np.zeros((height, width,3))
                input_dim = input_center.shape[-1]
                img_lA = torch.zeros(height, width, 3, input_dim, device=DEVICE)
                img_uA = torch.zeros(height, width, 3, input_dim, device=DEVICE)
                img_lbias = torch.zeros(height, width, 3, device=DEVICE)
                img_ubias = torch.zeros(height, width, 3, device=DEVICE)

            # Create Tiles Queue
            tiles_queue = [
                (h,w,min(h+tile_size, height),min(w+tile_size, width)) \
                for h in range(0, height, tile_size) for w in range(0, width, tile_size)
            ]

            if debug:
                pbar3 = tqdm(total=len(tiles_queue),desc="Processing Tiles", unit="item")

            while tiles_queue!=[]:
                hl,wl,hu,wu = tiles_queue.pop(0)
                tile_dict = {
                    "hl": hl,
                    "wl": wl,
                    "hu": hu,
                    "wu": wu,
                }

                # Crop Gaussians within Tile
                verf_net.call_model_preprocess("crop_gauss",input_center, tile_dict)

                if save_ref:
                    ref_tile = alpha_blending_ref(verf_net, input_center)
                    ref_tile_np = ref_tile.squeeze(0).detach().cpu().numpy()
                    img_ref[hl:hu, wl:wu, :] = ref_tile_np

                if save_bound:
                    lb_tile, ub_tile, lA_tile, uA_tile, lbias_tile, ubias_tile = alpha_blending_ptb(verf_net, input_center, input_lb, input_ub, bound_method)
                    lb_tile_np = lb_tile.squeeze(0).detach().cpu().numpy()
                    ub_tile_np = ub_tile.squeeze(0).detach().cpu().numpy()
                    img_lb[hl:hu, wl:wu, :] = lb_tile_np
                    img_ub[hl:hu, wl:wu, :] = ub_tile_np
                    img_lA[hl:hu, wl:wu, :, :] = lA_tile.squeeze(0)
                    img_uA[hl:hu, wl:wu, :, :] = uA_tile.squeeze(0)
                    img_lbias[hl:hu, wl:wu, :] = lbias_tile.squeeze(0)
                    img_ubias[hl:hu, wl:wu, :] = ubias_tile.squeeze(0)

                if debug:
                    pbar3.update(1)
            if debug:
                pbar3.close()

            if save_ref:
                img_ref= (img_ref.clip(min=0.0, max=1.0)*255).astype(np.uint8)
                res_ref = Image.fromarray(img_ref)
                res_ref.save(f'{save_folder_full}/ref_{absimg_num:06d}.png')

            if save_bound:
                img_lb_f = img_lb.clip(min=0.0, max=1.0).astype(np.float32, copy=False)
                img_ub_f = img_ub.clip(min=0.0, max=1.0).astype(np.float32, copy=False)
                _save_abstract_record_with_bounds(
                    save_dir=save_folder_full,
                    index=absimg_num,
                    lower_input=input_lb_org,
                    upper_input=input_ub_org,
                    lower_img=img_lb_f,
                    upper_img=img_ub_f,
                    img_lA=img_lA,
                    img_uA=img_uA,
                    img_lbias=img_lbias,
                    img_ubias=img_ubias,
                    point=base_trans,
                    direction=direction,
                    radius=radius,
                    lower_rel=lower_rel,
                    upper_rel=upper_rel,
                )

            absimg_num+=1
            pbar2.update(1)
        pbar2.close()

        pbar.update(1)
    pbar.close()

    return 0

if __name__ == '__main__':
    ### Default command: python3 scripts/abstract_new_gsplat.py --config configs/${case_name}/config.yaml --odd configs/${case_name}/traj.json
    parser = argparse.ArgumentParser(description="Abstract Gsplat with YAML configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--odd", type=str, required=True, help="Path to the traj.json file.")
    args = parser.parse_args()

    # Load parameters from YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    with open(args.odd, 'r') as file:
        odd_file = json.load(file)

    # Automatically determine scene_path and save_folder
    scene_path = f"../nerfstudio/outputs/{config['case_name']}/{config['render_method']}/{config['data_time']}"
    save_folder = f"../Outputs/AbstractImages/{config['case_name']}/{config['odd_type']}"

    # Determine perturbation type from traj.json (first entry), fallback to config or "cylinder"
    pert_type = odd_file[0].get("pert_type", config.get("pert_type", "cylinder"))

    # Load the correct radius field depending on pert_type
    if pert_type == "cuboid":
        # cuboidal_halflen is [x_half, z_half] per entry
        radiuss_list = [odd.get("cuboidal_halflen") for odd in odd_file]
    else:
        radiuss_list = [odd.get("radius") for odd in odd_file]

    downsampling_ratio = config["downsampling_ratio"]
    setup_dict = {
        "bound_method": config["bound_method"],
        "render_method": config["render_method"],
        "case_name": config["case_name"],
        "odd_type": config["odd_type"],
        "pert_type": pert_type,
        "debug": config["debug"],

        "width": int(config["width"]/downsampling_ratio),
        "height": int(config["height"]/downsampling_ratio),
        "fx": config["fx"]/downsampling_ratio,
        "fy": config["fy"]/downsampling_ratio,
        "eps2d": config["eps2d"]/downsampling_ratio,
        "tile_size": config["tile_size_abstract"],
        "min_distance": config["min_distance"],
        "max_distance": config["max_distance"],
        "gs_batch": config["gs_batch"],
        "part": config["part"],

        "scene_path": scene_path,
        "checkpoint_filename": config["checkpoint_filename"],
        "save_folder": save_folder,

        "bg_img_path": config["bg_img_path"],
        "bg_pure_color": config["bg_pure_color"],

        "save_ref": config["save_ref"],
        "save_bound": config["save_bound"],
        "N_samples": config["N_samples"],

        "poses":    [odd["pose"] for odd in odd_file],
        "radiuss":  radiuss_list,
        "gates":    [odd.get("gate") for odd in odd_file],
        "includes": [odd.get("include", True) for odd in odd_file],
    }

    start_time = time.time()
    main(setup_dict)
    end_time = time.time()

    print(f"Running Time: {(end_time - start_time) / 60:.4f} min")