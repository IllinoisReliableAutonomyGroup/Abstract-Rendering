import os 
import json 
import sys

import time
import numpy as np 
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt 
from tqdm import tqdm

from scipy.spatial.transform import Rotation 
from PIL import Image 

grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(grandfather_path)

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from collections import defaultdict

from utils import dir_to_rpy_and_rot, generate_samples, generate_trajectory,generate_single
from utils import convert_input_to_rot

from render_functions import GsplatRGBOrigin, TransferModelOrigin

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32
    
def main(setup_dict):
    key_list = ["bound_method", "render_method", "width", "height", "f", "tile_size", "partition_per_dim", "selection_per_dim", "scene_path", "checkpoint_filename", "bg_img_path", "save_folder", "save_ref", "save_bound", "domain_type", "N_samples", "input_min", "input_max","start_arr", "end_arr", "trans_arr"]
    bound_method, render_method, width, height, f, tile_size,  partition_per_dim, selection_per_dim, scene_path, checkpoint_filename, bg_img_path, save_folder, save_ref, save_bound, domain_type, N_samples, input_min, input_max,start_arr, end_arr, trans_arr = (setup_dict[key] for key in key_list)
    
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
        "fx": f,
        "fy": f,
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
        bg_pure_color = torch.tensor([123/255, 139/255, 196/255])
        bg_color = bg_pure_color.view(1, 1, 3).repeat(height, width,  1).to(DEVICE)
    else:
        bg_img = Image.open(bg_img_path).convert("RGB")  # ensure 3 channels
        bg_img = bg_img.resize((height, width), Image.LANCZOS) 
        bg_img = np.array(bg_img, dtype=np.float32)  # shape: (H, W, 3)
        bg_color = torch.from_numpy(bg_img/255).to(DEVICE)  # shape: (H, W, 3)
    
    # Generate Rotation Matrix
    
    rot = dir_to_rpy_and_rot(start_arr, end_arr)
    rot = torch.from_numpy(rot).to(dtype=DTYPE, device=DEVICE)
    # trans = np.array([-np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20)), 0.0])*6
    trans = torch.from_numpy(trans_arr).to(device=DEVICE, dtype=DTYPE)
    # print("rot:",rot)

    input_ref = (input_min + input_max)/2
    input_min, input_max, input_ref = input_min.unsqueeze(0), input_max.unsqueeze(0), input_ref.unsqueeze(0) #[1, N]
    inputs_ref = generate_single(input_min, input_max, input_ref)#generate_trajectory(input_min, input_max, input_ref, N_samples) # [N_samples+3, N]
    inputs_ref = inputs_ref[:].to(DEVICE)
    # partition_num = len(inputs_ref)
    

    inputs_queue = [input_ref for input_ref in inputs_ref] #list(zip(inputs_lb, inputs_ub, inputs_ref))
   
    absimg_num = 0

    # initialize tqdm without a fixed total
    pbar = tqdm(total=len(inputs_queue),desc="Processing inputs", unit="item")

    while inputs_queue:
        input_ref = inputs_queue.pop(0) # [N, ]
        input_ref = input_ref.unsqueeze(0) #[1, N]
        # print(input_lb, input_ub, input_ref)

        img_ref = np.zeros((height, width,3))

        rot = convert_input_to_rot(input_ref, trans, domain_type)
        rot = torch.from_numpy(rot).to(dtype=DTYPE, device=DEVICE)

        render_net = GsplatRGBOrigin(camera_dict, scene_dict_all, bg_color).to(DEVICE)
        verf_net = TransferModelOrigin(render_net, rot, trans, transform_hom, scale, domain_type).to(DEVICE)
        verf_net.call_model_preprocess("sort_gauss", input_ref)
        
        tiles_queue = [
            (h,w,min(h+tile_size, height),min(w+tile_size, width)) \
            for h in range(0, height, tile_size) for w in range(0, width, tile_size) 
        ] 

        while tiles_queue!=[]:
            hl,wl,hu,wu = tiles_queue.pop(0)
            tile_dict = {
                "hl": hl,
                "wl": wl,
                "hu": hu,
                "wu": wu,
            }

            #input_samples = generate_samples(input_lb, input_ub, input_ref)
            verf_net.call_model("update_tile", tile_dict)

            if save_ref:
                ref_tile = verf_net.forward(input_ref)#alpha_blending_ref(verf_net, input_ref)
                # print(f"ref_tile min and max: {torch.min(ref_tile).item():.4} {torch.max(ref_tile).item():.4}")
                ref_tile_np = ref_tile.squeeze(0).detach().cpu().numpy()
                img_ref[hl:hu, wl:wu, :] = ref_tile_np
                

        if save_ref:
            img_ref= (img_ref.clip(min=0.0, max=1.0)*255).astype(np.uint8)
            res_ref = Image.fromarray(img_ref)
            res_ref.save(f'{save_folder_full}/ref_{absimg_num}.png')

        absimg_num+=1

        pbar.update(1)
        # if absimg_num>=1:
        #     break
    pbar.close()

            
        

    return 0

if __name__=='__main__':

    # Setup Parameters
    bound_method = 'forward'
    render_method = 'gsplat_rgb'
    object_name = "airplane_grey"
    
    width = 64*2#80#
    height = 64*2#80#
    f = 80*2#100#
    tile_size = 64 #80

    partition_per_dim = 20000##5000
    selection_per_dim = 200

    scene_path = 'outputs/airplane_grey/splatfacto/2025-08-02_025446'
    checkpoint_filename = "step-000299999.ckpt"

    bg_img_path = None#"./BgImg/mountain.jpg"

    domain_type = "y"

    save_folder = "../Outputs/RenderedImages/"+object_name+"/"+domain_type
    save_ref = True
    save_bound = True

    N_samples = 30

    # input_min = torch.tensor([6, np.deg2rad(13), np.deg2rad(-1)]).to(DEVICE)
    # input_max = torch.tensor([7, np.deg2rad(15), np.deg2rad(1)]).to(DEVICE)
    # yaw
    # input_min = torch.tensor([-np.deg2rad(30)]).to(DEVICE)
    # input_max = torch.tensor([np.deg2rad(30)]).to(DEVICE)
    # y
    input_min = torch.tensor([-2]).to(DEVICE)
    input_max = torch.tensor([2]).to(DEVICE)
    # z and x
    # input_min = torch.tensor([-1]).to(DEVICE)
    # input_max = torch.tensor([1]).to(DEVICE)
    # round
    # input_min = torch.tensor([0]).to(DEVICE)
    # input_max = torch.tensor([2*np.pi-0.001]).to(DEVICE)

    setup_dict = {
        "bound_method": bound_method,
        "render_method": render_method,
        "width": width,
        "height": height,
        "f": f,
        "tile_size": tile_size,
        "partition_per_dim": partition_per_dim,
        "selection_per_dim": selection_per_dim,
        "scene_path": scene_path,
        "checkpoint_filename": checkpoint_filename,
        "bg_img_path": bg_img_path,
        "save_folder": save_folder,
        "save_ref": save_ref,
        "save_bound": save_bound,
        "domain_type": domain_type,
        "N_samples": N_samples,
        "input_min": input_min,
        "input_max": input_max,
        "start_arr": np.array([-np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20)), 0.0])*6,
        "end_arr": np.array([0.0, 0.0, 0.0]),
        "trans_arr": np.array([-np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20)), 0.0])*6,
    }

    start_time=time.time()
    main(setup_dict)
    end_time = time.time()

    print(f"Running Time:{(end_time-start_time)/60:.4f} min")


