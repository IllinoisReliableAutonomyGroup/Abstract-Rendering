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

from scripts.utils_partition import generate_partition
from utils_operation import alpha_blending, alpha_blending_interval
from render_models import GsplatRGB, TransferModel

import argparse

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

bound_opts = {
    'conv_mode': 'matrix',
    'optimize_bound_args': {
        'iteration': 100, 
        # 'lr_alpha':0.02, 
        'early_stop_patience':5},
} 

# --- Drop-in: helper to save abstract record (.pt with 8 fields)
def save_abstract_record(save_dir, index, lower_input, upper_input, lower_img, upper_img):
    """
    Save an abstract image record with required fields:
      lower, upper, lA, uA, lb, ub, xl, xu
    """

    if isinstance(lower_input, np.ndarray):
        lower_i = torch.from_numpy(lower_input.astype(np.float32, copy=False))
    else:
        lower_i = lower_input.to(dtype=torch.float32).detach().cpu()

    if isinstance(upper_input, np.ndarray):
        upper_i = torch.from_numpy(upper_input.astype(np.float32, copy=False))
    else:
        upper_i = upper_input.to(dtype=torch.float32).detach().cpu()

    if isinstance(lower_img, np.ndarray):
        lower_t = torch.from_numpy(lower_img.astype(np.float32))
    else:
        lower_t = lower_img.to(dtype=torch.float32).cpu()

    if isinstance(upper_img, np.ndarray):
        upper_t = torch.from_numpy(upper_img.astype(np.float32))
    else:
        upper_t = upper_img.to(dtype=torch.float32).detach().cpu()
    
    record = {
        "lower": lower_t,  # (H, W, 3), float32, [0,1]
        "upper": upper_t,  # (H, W, 3), float32, [0,1]
        "lA": None,
        "uA": None,
        "lb": None,
        "ub": None,
        "xl": lower_i,
        "xu": upper_i,
    }
    out_path = os.path.join(save_dir, f"abstract_{index:06d}.pt")
    torch.save(record, out_path)
    return out_path


def alpha_blending_ref(net, input_ref):
    
    N = net.call_model("get_num")
    triu_mask = torch.triu(torch.ones(N+2, N+2), diagonal=1)
    bg_color=(net.call_model("get_bg_color_tile")).unsqueeze(0).unsqueeze(-2) #[1, TH, TW, N, 3]

    #print(f"Number of Gaussians used in rendering: {N}")
    if N==0:
        return bg_color.squeeze(-2)

    else:
        net.call_model("update_model_param", 0,N,"fast")
        colors_alpha = net.call_model_preprocess("render_color_alpha", input_ref)  #[1, TH, TW, N, 4]

        colors, alpha = colors_alpha.split([3,1], dim=-1)

        ones = torch.ones_like(alpha[:, :, :, 0:1, :])
        alpha = torch.cat([alpha,ones], dim=-2) # [1, TH, TW, 2, 1]
        colors = torch.cat([colors,bg_color], dim=-2) # [1, TH, TW, 2, 3]

        colors_alpha_out = alpha_blending(alpha, colors, "fast", triu_mask)
        color_out, alpha_out = colors_alpha_out.split([3,1], dim=-1)

        color_out = color_out.squeeze(-2)
        return color_out


def alpha_blending_ptb(net, input_ref, input_lb, input_ub, bound_method):
    N = net.call_model("get_num")
    gs_batch = net.call_model("get_gs_batch")
    bg_color=(net.call_model("get_bg_color_tile")).unsqueeze(0).unsqueeze(-2) #[1, TH, TW, N, 3]

    if N==0:
        return bg_color.squeeze(-2), bg_color.squeeze(-2)
    else:
        alphas_int_lb = []
        alphas_int_ub = []

        hl,wl,hu,wu = (net.call_model("get_tile_dict")[key] for key in ["hl", "wl", "hu", "wu"])

        ptb = PerturbationLpNorm(x_L=input_lb,x_U=input_ub)
        input_ptb = BoundedTensor(input_ref, ptb)

        with torch.no_grad():
            for i, idx_start in enumerate(range(0, N, gs_batch)):
                idx_end = min(idx_start + gs_batch, N)

                net.call_model("update_model_param",idx_start,idx_end,"middle")
                model = BoundedModule(net, input_ref, bound_opts=bound_opts, device=DEVICE)

                alpha_ibp_lb, alpha_ibp_ub = model.compute_bounds(x=(input_ptb, ), method="ibp")
                reference_interm_bounds = {}
                for node in model.nodes():
                    if (node.perturbed
                        and isinstance(node.lower, torch.Tensor)
                        and isinstance(node.upper, torch.Tensor)):
                        reference_interm_bounds[node.name] = (node.lower, node.upper)

                # required_A = defaultdict(set)
                # required_A[model.output_name[0]].add(model.input_name[0])

                alpha_int_lb, alpha_int_ub= model.compute_bounds(
                    x= (input_ptb, ), 
                    method=bound_method, 
                    reference_bounds=reference_interm_bounds, 
                )  #[1, TH, TW, N, 4]
                
                # lower_A, lower_bias = A_dict[model.output_name[0]][model.input_name[0]]['lA'], A_dict[model.output_name[0]][model.input_name[0]]['lbias']
                # upper_A, upper_bias = A_dict[model.output_name[0]][model.input_name[0]]['uA'], A_dict[model.output_name[0]][model.input_name[0]]['ubias']
                # print(f"lower_A shape: {lower_A.shape}, lower_bias shape: {lower_bias.shape}")
                # print(f"upper_A shape: {upper_A.shape}, upper_bias shape: {upper_bias.shape}")
        
                alpha_int_lb = alpha_int_lb.reshape(1, hu-hl, wu-wl, idx_end-idx_start, 1)
                alpha_int_ub = alpha_int_ub.reshape(1, hu-hl, wu-wl, idx_end-idx_start, 1)

                alphas_int_lb.append(alpha_int_lb.detach())
                alphas_int_ub.append(alpha_int_ub.detach())

            del model
            torch.cuda.empty_cache()

            alphas_int_lb = torch.cat(alphas_int_lb, dim=-2)
            alphas_int_ub = torch.cat(alphas_int_ub, dim=-2)

        # Load Colors within Tile and Add background
        colors = net.call_model("get_color_tile")
        colors = colors.view(1, 1, 1, alphas_int_lb.size(-2), 3).repeat(1, alpha_int_lb.size(1), alpha_int_lb.size(2), 1, 1)
        colors = torch.cat([colors, bg_color], dim = -2)

        ones = torch.ones_like(alphas_int_lb[:, :, :, 0:1, :])
        alphas_int_lb = torch.cat([alphas_int_lb, ones], dim=-2)
        alphas_int_ub = torch.cat([alphas_int_ub, ones], dim=-2)        

        color_alpha_out_lb, color_alpha_out_ub = alpha_blending_interval(alphas_int_lb, alphas_int_ub, colors)

        color_out_lb,alpha_out_lb = color_alpha_out_lb.split([3,1],dim=-1)
        color_out_ub,alpha_out_ub = color_alpha_out_ub.split([3,1],dim=-1)

    return color_out_lb.squeeze(-2), color_out_ub.squeeze(-2)

    
def main(setup_dict):
    ### Default command: python3 scripts/abstract_gsplat.py --config configs/uturn/config.yaml --odd configs/uturn/traj.json
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
    
    # Generate Rotation Matrix
    transs, orientations = np.array(poses)[:, :3], np.array(poses)[:, 3:]
    Rs = Rotation.from_euler('ZYX', orientations)
    radiuss = np.array(radiuss)
    rots = Rs.as_matrix()

    # Create Queue for Inputs
    queue = deque(zip(transs, rots, radiuss))
    absimg_num = 0
    pbar = tqdm(total=len(queue)-1,desc="Processing Traj", unit="item")

    # Define Render and Verification Network
    pipeline_type = "abstract-rendering"
    render_net = GsplatRGB(camera_dict, scene_dict_all, min_distance, max_distance, bg_color, eps2d, gs_batch).to(DEVICE)
    verf_net = TransferModel(render_net, pipeline_type, None, None, transform_hom, scale, odd_type).to(DEVICE)

    # Process Each Input
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

        inputs_queue = deque(zip(inputs_center, inputs_lb, inputs_ub))
        pbar2 = tqdm(total=len(inputs_queue),desc="Processing Inputs", unit="item")

        while inputs_queue:
            input_center, input_lb, input_ub = inputs_queue.popleft() # [N, ]
            input_center, input_lb, input_ub = input_center.unsqueeze(0), input_lb.unsqueeze(0), input_ub.unsqueeze(0) #[1, N]
            verf_net.call_model_preprocess("sort_gauss", input_center)

            if save_ref:
                img_ref = np.zeros((height, width,3))
            if save_bound:
                img_lb = np.zeros((height, width,3))
                img_ub = np.zeros((height, width,3))
        
            
            tiles_queue = [
                (h,w,min(h+tile_size, height),min(w+tile_size, width)) \
                for h in range(0, height, tile_size) for w in range(0, width, tile_size) 
            ] 

            pbar3 = tqdm(total=len(tiles_queue),desc="Processing Tiles", unit="item")

            while tiles_queue!=[]:
                hl,wl,hu,wu = tiles_queue.pop(0)
                tile_dict = {
                    "hl": hl,
                    "wl": wl,
                    "hu": hu,
                    "wu": wu,
                }

                # #print(f"Processing absimg {absimg_num:06d}, tile hl:{hl}, wl:{wl}, hu:{hu}, wu:{wu}")
                # input_samples = generate_samples(input_lb, input_ub, input_ref, N_samples)
                # verf_net.call_model_preprocess("crop_gauss",input_samples, tile_dict)
                verf_net.call_model_preprocess("crop_gauss",input_center, tile_dict)

                if save_ref:
                    ref_tile = alpha_blending_ref(verf_net, input_center)
                    # print(f"ref_tile min and max: {torch.min(ref_tile).item():.4} {torch.max(ref_tile).item():.4}")
                    ref_tile_np = ref_tile.squeeze(0).detach().cpu().numpy()
                    img_ref[hl:hu, wl:wu, :] = ref_tile_np

                if save_bound:
                    lb_tile, ub_tile = alpha_blending_ptb(verf_net, input_center, input_lb, input_ub, bound_method)
                    # print(f"lb_tile min and ub_tile max: {torch.min(lb_tile).item():.4} {torch.max(ub_tile).item():.4}")
                    lb_tile_np = lb_tile.squeeze(0).detach().cpu().numpy() # [TH, TW, 3]
                    ub_tile_np = ub_tile.squeeze(0).detach().cpu().numpy()
                    img_lb[hl:hu, wl:wu, :] = lb_tile_np
                    img_ub[hl:hu, wl:wu, :] = ub_tile_np

                    pbar3.update(1)
            pbar3.close()

            if save_ref:
                img_ref= (img_ref.clip(min=0.0, max=1.0)*255).astype(np.uint8)
                res_ref = Image.fromarray(img_ref)
                res_ref.save(f'{save_folder_full}/ref_{absimg_num:06d}.png')

            if save_bound:
                # --- Drop-in replacement: save .pt record with 8 fields instead of PNGs
                img_lb_f = img_lb.clip(min=0.0, max=1.0).astype(np.float32, copy=False)
                img_ub_f = img_ub.clip(min=0.0, max=1.0).astype(np.float32, copy=False)
                # print("input_lb:", input_lb)
                save_abstract_record(
                    save_dir=save_folder_full,
                    index=absimg_num,
                    lower_input = input_lb,
                    upper_input=input_ub,
                    lower_img=img_lb_f,
                    upper_img=img_ub_f,
                )

            absimg_num+=1
            pbar2.update(1)
        pbar2.close()

        pbar.update(1)
    pbar.close()

    return 0

if __name__ == '__main__':
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
    scene_path = f"outputs/{config['object_name']}/{config['render_method']}/{config['data_time']}"
    save_folder = f"../Outputs/AbstractImages/{config['object_name']}/{config['odd_type']}"

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
