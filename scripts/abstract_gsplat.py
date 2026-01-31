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
from render_models import GsplatRGB, TransferModel

import argparse

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32


    
def main(setup_dict):
    key_list = [
        "bound_method", "render_method", "case_name", "odd_type", "debug", "width", "height",
        "fx", "fy", "eps2d", "tile_size", "min_distance", "max_distance", "gs_batch",
        "part", "scene_path", "checkpoint_filename",
        "save_folder", "bg_img_path", "bg_pure_color", "save_ref", "save_bound",
        "N_samples", "poses", "radiuss"
    ]

    bound_method, render_method, case_name, odd_type, debug, width, height, fx, fy, eps2d, \
    tile_size, min_distance, max_distance, gs_batch, part, \
    scene_path, checkpoint_filename, save_folder, bg_img_path, \
    bg_pure_color, save_ref, save_bound, N_samples, poses, radiuss = itemgetter(*key_list)(setup_dict)

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
    scene_parameters = torch.load(checkpoint_file)
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

    # Create Queue for Waypoints
    queue = deque(zip(transs, rots, radiuss))
    absimg_num = 0
    pbar = tqdm(total=len(queue)-1,desc="Processing Poses", unit="item")

    # Define Render and Verification Network
    pipeline_type = "abstract-rendering"
    render_net = GsplatRGB(camera_dict, scene_dict_all, min_distance, max_distance, bg_color, eps2d, gs_batch).to(DEVICE)
    verf_net = TransferModel(render_net, pipeline_type, None, None, transform_hom, scale, odd_type).to(DEVICE)

    # Create Partition Inputs   
    inputs_center, inputs_lb, inputs_ub = generate_partition(odd_type, part) # [part, N]
    inputs_center, inputs_lb, inputs_ub= torch.tensor(inputs_center).to(device=DEVICE, dtype=DTYPE), torch.tensor(inputs_lb).to(device=DEVICE, dtype=DTYPE), torch.tensor(inputs_ub).to(device=DEVICE, dtype=DTYPE)
    # partition_num = len(inputs_center)

    while len(queue)>1:
        base_trans, rot, radius = queue.popleft() # [N, ]
        base_trans_next = queue[0][0]

        base_trans = torch.from_numpy(base_trans).to(device=DEVICE, dtype=DTYPE) #[1, 3]
        base_trans_next = torch.from_numpy(base_trans_next).to(device=DEVICE, dtype=DTYPE)
        direction = base_trans_next - base_trans
        rot = torch.from_numpy(rot).to(device=DEVICE, dtype=DTYPE) #[3, 3]  
        radius = torch.tensor(radius, device=DEVICE, dtype=DTYPE) #[1, ]
        verf_net.update_model_param(rot, base_trans, direction, radius)

        # Create Queue for Inputs (Pose Cells)
        inputs_queue = deque(zip(inputs_center, inputs_lb, inputs_ub))
        pbar2 = tqdm(total=len(inputs_queue),desc="Processing Cells", unit="item")

        while inputs_queue:
            input_center_org, input_lb_org, input_ub_org = inputs_queue.popleft() # [N, ]
            input_center, input_lb, input_ub = refine_partition(input_center_org.clone(), input_lb_org.clone(), input_ub_org.clone(), odd_type) 
            input_center, input_lb, input_ub = input_center.unsqueeze(0), input_lb.unsqueeze(0), input_ub.unsqueeze(0) #[1, N]
            
            # Sort Gaussians based on Distance to Camera
            verf_net.call_model_preprocess("sort_gauss", input_center)

            if save_ref:
                img_ref = np.zeros((height, width,3))
            if save_bound:
                img_lb = np.zeros((height, width,3))
                img_ub = np.zeros((height, width,3))
        
            # Create Tiles Queue
            tiles_queue = [
                (h,w,min(h+tile_size, height),min(w+tile_size, width)) \
                for h in range(0, height, tile_size) for w in range(0, width, tile_size) 
            ] 

            if debug:
                pbar3 = tqdm(total=len(tiles_queue),desc="Processing Tiles", unit="item", disable=True)

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
                    lb_tile, ub_tile = alpha_blending_ptb(verf_net, input_center, input_lb, input_ub, bound_method)
                    lb_tile_np = lb_tile.squeeze(0).detach().cpu().numpy() # [TH, TW, 3]
                    ub_tile_np = ub_tile.squeeze(0).detach().cpu().numpy()
                    img_lb[hl:hu, wl:wu, :] = lb_tile_np
                    img_ub[hl:hu, wl:wu, :] = ub_tile_np

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
                save_abstract_record(
                    save_dir=save_folder_full,
                    index = absimg_num,
                    lower_input = input_lb_org,
                    upper_input = input_ub_org,
                    lower_img=img_lb_f,
                    upper_img=img_ub_f,
                    point = base_trans,
                    direction = direction,
                    radius = radius,
                )

            absimg_num+=1
            pbar2.update(1)
        pbar2.close()

        pbar.update(1)
    pbar.close()

    return 0

if __name__ == '__main__':
    ### Default command: python3 scripts/abstract_gsplat.py --config configs/${case_name}/config.yaml --odd configs/${case_name}/traj.json
    parser = argparse.ArgumentParser(description="Abstract Gsplat with YAML configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--odd", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load parameters from YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    with open(args.odd, 'r') as file:
        odd_file = json.load(file)

    # Automatically determine scene_path and save_folder
    scene_path = f"../nerfstudio/outputs/{config['case_name']}/{config['render_method']}/{config['data_time']}"
    save_folder = f"../Outputs/AbstractImages/{config['case_name']}/{config['odd_type']}"

    downsampling_ratio = config["downsampling_ratio"]
    setup_dict = {
        "bound_method": config["bound_method"],
        "render_method": config["render_method"],
        "case_name": config["case_name"],
        "odd_type": config["odd_type"],
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
        
        "poses": [odd["pose"] for odd in odd_file],
        "radiuss": [odd["radius"] if "radius" in odd else None for odd in odd_file],
    }

    start_time = time.time()
    main(setup_dict)
    end_time = time.time()

    print(f"Running Time: {(end_time - start_time) / 60:.4f} min")
