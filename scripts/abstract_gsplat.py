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

from utils import dir_to_rpy_and_rot, convert_input_to_rot
from utils import generate_bound,  generate_samples
from utils import alpha_blending, alpha_blending_interval
from render_models import GsplatRGB, TransferModel



# from simple_model2_alphatest5_2 import AlphaModel, DepthModel, MeanModel
# from rasterization_pytorch import rasterize_gaussians_pytorch_rgb
# from generate_poses import generate_poses

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
}, 

# --- Drop-in: helper to save abstract record (.pt with 8 fields)
def save_abstract_record(save_dir, index, lower_img, upper_img):
    """
    Save an abstract image record with required fields:
      lower, upper, lA, uA, lb, ub, xl, xu
    Currently only lower/upper are populated; others are left None.
    lower_img / upper_img are expected as arrays/tensors in [0,1] with shape (H,W,3).
    """
    if isinstance(lower_img, np.ndarray):
        lower_t = torch.from_numpy(lower_img.astype(np.float32, copy=False))
    else:
        lower_t = lower_img.to(dtype=torch.float32).detach().cpu()

    if isinstance(upper_img, np.ndarray):
        upper_t = torch.from_numpy(upper_img.astype(np.float32, copy=False))
    else:
        upper_t = upper_img.to(dtype=torch.float32).detach().cpu()

    record = {
        "lower": lower_t,  # (H, W, 3), float32, [0,1]
        "upper": upper_t,  # (H, W, 3), float32, [0,1]
        "lA": None,
        "uA": None,
        "lb": None,
        "ub": None,
        "xl": None,
        "xu": None,
    }
    out_path = os.path.join(save_dir, f"abstract_{index:06d}.pt")
    torch.save(record, out_path)
    return out_path


def alpha_blending_ref(net, input_ref):
    
    N = net.call_model("get_num")
    triu_mask = torch.triu(torch.ones(N+2, N+2), diagonal=1)
    bg_color=(net.call_model("get_bg_color_tile")).unsqueeze(0).unsqueeze(-2) #[1, TH, TW, N, 3]

    if N==0:
        return bg_color.squeeze(-2)

    else:
        # N=min(N,2000)
        # net.call_model("update_model_param", 0,N,"middle")
        # model = BoundedModule(net, input_ref, device=DEVICE)
        # colors_alpha = model.forward(input_ref)  #[1, TH, TW, N, 4]

        net.call_model("update_model_param", 0,N,"fast")
        # print("intpu_ref:", input_ref.shape)
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
                # print("epoch:", i)

                net.call_model("update_model_param",idx_start,idx_end,"middle")
                model = BoundedModule(net, input_ref, bound_opts=bound_opts, device=DEVICE)

                alpha_ibp_lb, alpha_ibp_ub = model.compute_bounds(x=(input_ptb, ), method="ibp")
                reference_interm_bounds = {}
                for node in model.nodes():
                    if (node.perturbed
                        and isinstance(node.lower, torch.Tensor)
                        and isinstance(node.upper, torch.Tensor)):
                        reference_interm_bounds[node.name] = (node.lower, node.upper)

                alpha_int_lb, alpha_int_ub = model.compute_bounds(x= (input_ptb, ), method="forward", reference_bounds=reference_interm_bounds)  #[1, TH, TW, N, 4]
                
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
    key_list = ["bound_method", "render_method", "width", "height", "f", "tile_size", "partition_per_dim", "selection_per_dim", "scene_path", "checkpoint_filename", "bg_img_path", "save_folder", "save_ref", "save_bound", "domain_type", "N_samples", "input_min", "input_max","start_arr", "end_arr", "trans_arr"]
    bound_method, render_method, width, height, f, tile_size,  partition_per_dim, selection_per_dim, scene_path, checkpoint_filename, bg_img_path, save_folder, save_ref, save_bound, domain_type, N_samples, input_min, input_max, start_arr, end_arr, trans_arr = (setup_dict[key] for key in key_list)
    
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
    # start_arr = np.array([-np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20)), 0.0])*6
    # end_arr = np.array([0.0, 0.0, 0.0])
    rot = dir_to_rpy_and_rot(start_arr, end_arr)
    rot = torch.from_numpy(rot).to(dtype=DTYPE, device=DEVICE)
    # trans = np.array([-np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20)), 0.0])*6
    trans = torch.from_numpy(trans_arr).to(device=DEVICE, dtype=DTYPE)
    # print("rot:",rot)

    inputs_lb, inputs_ub, inputs_ref = generate_bound(input_min, input_max, partition_per_dim, selection_per_dim) # [partition_per_dim^N, N]
    inputs_lb, inputs_ub, inputs_ref = inputs_lb.to(DEVICE), inputs_ub.to(DEVICE), inputs_ref.to(DEVICE)
    # partition_num = len(inputs_ref)
    

    inputs_queue = list(zip(inputs_lb, inputs_ub, inputs_ref))

    absimg_num = 0

    # initialize tqdm without a fixed total
    pbar = tqdm(total=len(inputs_queue),desc="Processing inputs", unit="item")

    while inputs_queue:
        input_lb, input_ub, input_ref = inputs_queue.pop(0) # [N, ]
        input_lb, input_ub, input_ref = input_lb.unsqueeze(0), input_ub.unsqueeze(0), input_ref.unsqueeze(0) #[1, N]
        # print(input_lb, input_ub, input_ref)

        # ptb = PerturbationLpNorm(x_L=input_lb,x_U=input_ub)
        # input_ptb = BoundedTensor(input_ref, ptb)

        if save_ref:
            img_ref = np.zeros((height, width,3))
        if save_bound:
            img_lb = np.zeros((height, width,3))
            img_ub = np.zeros((height, width,3))

        rot = convert_input_to_rot(input_ref, trans, domain_type)
        rot = torch.from_numpy(rot).to(dtype=DTYPE, device=DEVICE)

        render_net = GsplatRGB(camera_dict, scene_dict_all, bg_color).to(DEVICE)
        verf_net = TransferModel(render_net, rot, trans, transform_hom, scale, domain_type).to(DEVICE)
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

            input_samples = generate_samples(input_lb, input_ub, input_ref, N_samples)
            verf_net.call_model_preprocess("crop_gauss",input_samples, tile_dict)

            if save_ref:
                ref_tile = alpha_blending_ref(verf_net, input_ref)
                # print(f"ref_tile min and max: {torch.min(ref_tile).item():.4} {torch.max(ref_tile).item():.4}")
                ref_tile_np = ref_tile.squeeze(0).detach().cpu().numpy()
                img_ref[hl:hu, wl:wu, :] = ref_tile_np

            if save_bound:
                lb_tile, ub_tile = alpha_blending_ptb(verf_net, input_ref, input_lb, input_ub, bound_method)
                # print(f"lb_tile min and ub_tile max: {torch.min(lb_tile).item():.4} {torch.max(ub_tile).item():.4}")
                lb_tile_np = lb_tile.squeeze(0).detach().cpu().numpy() # [TH, TW, 3]
                ub_tile_np = ub_tile.squeeze(0).detach().cpu().numpy()
                img_lb[hl:hu, wl:wu, :] = lb_tile_np
                img_ub[hl:hu, wl:wu, :] = ub_tile_np

            
        if save_ref:
            img_ref= (img_ref.clip(min=0.0, max=1.0)*255).astype(np.uint8)
            res_ref = Image.fromarray(img_ref)
            res_ref.save(f'{save_folder_full}/ref_{absimg_num}.png')

        if save_bound:
            # --- Drop-in replacement: save .pt record with 8 fields instead of PNGs
            img_lb_f = img_lb.clip(min=0.0, max=1.0).astype(np.float32, copy=False)
            img_ub_f = img_ub.clip(min=0.0, max=1.0).astype(np.float32, copy=False)
            save_abstract_record(
                save_dir=save_folder_full,
                index=absimg_num,
                lower_img=img_lb_f,
                upper_img=img_ub_f,
            )

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
    tile_size = 6*2 #4

    partition_per_dim = 20000##5000
    selection_per_dim = 200

    scene_path = 'outputs/airplane_grey/splatfacto/2025-08-02_025446'
    checkpoint_filename = "step-000299999.ckpt"

    bg_img_path = None#"./BgImg/mountain.jpg"

    domain_type = "round"

    save_folder = "../Outputs/AbstractImages/"+object_name+"/"+domain_type
    save_ref = True
    save_bound = True

    N_samples = 5

    # input_min = torch.tensor([6, np.deg2rad(13), np.deg2rad(-1)]).to(DEVICE)
    # input_max = torch.tensor([7, np.deg2rad(15), np.deg2rad(1)]).to(DEVICE)
    # yaw
    # input_min = torch.tensor([-np.deg2rad(30)]).to(DEVICE)
    # input_max = torch.tensor([np.deg2rad(30)]).to(DEVICE)
    # y
    # input_min = torch.tensor([-2]).to(DEVICE)
    # input_max = torch.tensor([2]).to(DEVICE)
    # z and x
    # input_min = torch.tensor([-1]).to(DEVICE)
    # input_max = torch.tensor([1]).to(DEVICE)
    # round
    input_min = torch.tensor([0]).to(DEVICE)
    input_max = torch.tensor([2*np.pi-0.001]).to(DEVICE)

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
