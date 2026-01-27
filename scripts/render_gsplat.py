import os 
import json 
import sys
import argparse
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

from render_models import GsplatRGBOrigin, TransferModel

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32
    
def main(setup_dict):

    key_list = [
        "bound_method", "render_method", "object_name", "odd_type", "width", "height",
        "fx", "fy", "eps2d", "tile_size", "min_distance", "max_distance", "gs_batch",
        "part", "scene_path", "checkpoint_filename",
        "save_folder", "bg_img_path", "bg_pure_color", "save_ref", "save_bound",
        "N_samples", "poses", "radiuss"
    ]

    bound_method, render_method, object_name, odd_type, width, height, fx, fy, eps2d, \
    tile_size, min_distance, max_distance, gs_batch, part, \
    scene_path, checkpoint_filename, save_folder, bg_img_path, \
    bg_pure_color, save_ref, save_bound, N_samples, poses, radiuss = itemgetter(*key_list)(setup_dict)

    # Load Already Trained Scene Files
    script_dir = os.path.dirname(os.path.realpath(__file__))
    scene_folder = os.path.join(script_dir, '../nerfstudio/', scene_path)
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

    # Define camera_dict and scene_dict
    camera_dict = {
        "fx": fx,
        "fy": fy,
        "width": width,
        "height": height,
    }
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
    
    # Prepare Poses
    transs, orientations = np.array(poses)[:, :3], np.array(poses)[:, 3:]
    Rs = Rotation.from_euler('ZYX', orientations)
    rots = Rs.as_matrix()

    # Create Queue for Inputs
    queue = deque(zip(transs, rots))
    absimg_num = 0
    pbar = tqdm(total=len(queue),desc="Processing inputs", unit="item")
    
    # Define Render and Verification Network
    pipeline_type = "rendering"
    render_net = GsplatRGBOrigin(camera_dict, scene_dict_all, min_distance, max_distance, bg_color, eps2d).to(DEVICE)
    verf_net = TransferModel(render_net, pipeline_type, None, None, transform_hom, scale, odd_type).to(DEVICE)

    while queue:
        base_trans, rot = queue.popleft() # [N, ]
        base_trans = torch.from_numpy(base_trans).to(device=DEVICE, dtype=DTYPE) #[1, 3]
        rot = torch.from_numpy(rot).to(device=DEVICE, dtype=DTYPE) #[3, 3]  
        verf_net.update_model_param(rot, base_trans)

        if save_ref:
            img_ref = np.zeros((height, width,3))

        if odd_type=="cylinder":
            input_ref = torch.zeros((1,3)).to(device=DEVICE, dtype=DTYPE)

        verf_net.call_model_preprocess("sort_gauss", input_ref)
        
        tiles_queue = deque(
            (h, w, min(h + tile_size, height), min(w + tile_size, width))
            for h in range(0, height, tile_size)
            for w in range(0, width, tile_size)
        )

        while tiles_queue:
            hl, wl, hu, wu = tiles_queue.popleft()

            tile_dict = {"hl": hl, "wl": wl, "hu": hu, "wu": wu}

            #input_samples = generate_samples(input_lb, input_ub, input_ref)
            verf_net.call_model("update_tile", tile_dict)

            if save_ref:
                ref_tile = verf_net.forward(input_ref)
                # print(f"ref_tile min and max: {torch.min(ref_tile).item():.4} {torch.max(ref_tile).item():.4}")
                ref_tile_np = ref_tile.squeeze(0).detach().cpu().numpy()
                img_ref[hl:hu, wl:wu, :] = ref_tile_np
                

        if save_ref:
            img_ref= (img_ref.clip(min=0.0, max=1.0)*255).astype(np.uint8)
            res_ref = Image.fromarray(img_ref)
            res_ref.save(f'{save_folder_full}/ref_{absimg_num:06d}.png')
            print(f'{save_folder_full}/ref_{absimg_num:06d}.png')
  

        absimg_num+=1

        pbar.update(1)
    pbar.close()
    return 0

if __name__=='__main__':
    ### default command: python3 scripts/render_gsplat.py --config configs/uturn/config.yaml --odd configs/uturn/traj.json
    parser = argparse.ArgumentParser(description="Render Gsplat with YAML configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--odd", type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # Load parameters from YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    with open(args.odd, 'r') as file:
        odd_file = json.load(file)

    # Automatically determine scene_path and save_folder
    scene_path = f"outputs/{config['object_name']}/{config['render_method']}/{config['data_time']}"
    save_folder = f"../Outputs/RenderedImages/{config['object_name']}/{config['odd_type']}"
    if config['save_filename'] is not None:
        save_folder = os.path.join(save_folder, config['save_filename'])

    setup_dict = {
        "bound_method": config["bound_method"],
        "render_method": config["render_method"],
        "object_name": config["object_name"],
        "odd_type": config["odd_type"],

        "width": config["width"],
        "height": config["height"],
        "fx": config["fx"],
        "fy": config["fy"],
        "eps2d": config["eps2d"],
        "tile_size": config["tile_size_render"],
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


