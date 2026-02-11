import os
import sys
import torch

from utils_operation import regulate, cumprod

grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(grandfather_path)

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from collections import defaultdict



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

bound_opts = {
    'conv_mode': 'matrix',
    'optimize_bound_args': {
        'iteration': 100, 
        # 'lr_alpha':0.02, 
        'early_stop_patience':5},
} 

def alpha_blending(alpha, colors, method, triu_mask=None):

    N = alpha.size(-2)
    alpha = regulate(alpha)
    colors = regulate(colors)

    if method == 'fast':
        #transmittance = self.regulate(self.cumprod(1-alpha))
        alpha_shifted = torch.cat([torch.zeros_like(alpha[:,:,:,0:1,:], dtype=alpha.dtype), alpha[:,:,:,:-1,:]], dim=-2)
        transmittance = regulate(torch.cumprod((1-alpha_shifted), dim=-2))

        alpha_combined= regulate((alpha*transmittance).sum(dim=-2, keepdim=True)) # [1, TH, TW, 1, 1]
        colors_combined = regulate((alpha*transmittance*colors).sum(dim=-2, keepdim=True)) # [1, TH, TW, 1, 3]

    elif method == 'middle':
        transmittance = regulate(cumprod((1-alpha),triu_mask, dim=-2))

        alpha_combined= regulate(regulate(alpha*transmittance).sum(dim=-2, keepdim=True)) # [1, TH, TW, 1, 1]
        colors_combined = regulate(regulate(alpha*transmittance*colors).sum(dim=-2, keepdim=True)) # [1, TH, TW, 1, 3]

    elif method == 'slow':
        rgb_color = regulate(alpha*colors) # [1, TH, TW, N, 3]
        one_minus_alpha = regulate(1-alpha) # [1, TH, TW, N, 3]

        alpha_combined = torch.zeros_like(alpha[:, :, :, 0:1, :]) # [1, TH, TW, 1, 1]
        colors_combined = torch.zeros_like(colors[:, :, :, 0:1, :]) # [1, TH, TW, 1, 3]
        for i in range(N-1, -1, -1):
            alpha_combined = regulate(alpha[:, :, :, i:i+1, :]+one_minus_alpha[:, :, :, i:i+1, :]*alpha_combined) # [1, TH, TW, 1, 1]
            colors_combined = regulate(rgb_color[:, :, :, i:i+1, :]+one_minus_alpha[:, :, :, i:i+1, :]*colors_combined) # [1, TH, TW, 1, 3]

    colors_alpha_combined = torch.cat((colors_combined, alpha_combined), dim =-1)
    return colors_alpha_combined

def alpha_blending_interval(alpha_lb, alpha_ub, colors):
    
    alpha_lb_shifted = torch.cat([torch.zeros_like(alpha_lb[:,:,:,0:1,:], dtype=alpha_lb.dtype), alpha_lb[:,:,:,:-1,:]], dim=-2)
    transmittance_ub = regulate(torch.cumprod((1-alpha_lb_shifted), dim=-2))

    alpha_ub_shifted = torch.cat([torch.zeros_like(alpha_ub[:,:,:,0:1,:], dtype=alpha_lb.dtype), alpha_ub[:,:,:,:-1,:]], dim=-2)
    transmittance_lb = regulate(torch.cumprod((1-alpha_ub_shifted), dim=-2))

    alpha_out_lb = regulate(torch.sum((alpha_lb*transmittance_lb), dim=-2, keepdim=True))
    alpha_out_ub = regulate(torch.sum((alpha_ub*transmittance_ub), dim=-2, keepdim=True))

    color_out_lb = regulate(torch.sum((alpha_lb*transmittance_lb*colors), dim=-2, keepdim=True))
    color_out_ub = regulate(torch.sum((alpha_ub*transmittance_ub*colors), dim=-2, keepdim=True))

    color_alpha_out_lb = torch.cat([color_out_lb,alpha_out_lb], dim = -1)
    color_alpha_out_ub = torch.cat([color_out_ub,alpha_out_ub], dim = -1)
    return color_alpha_out_lb, color_alpha_out_ub


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


def alpha_blending_ptb(net, input_ref, input_lb, input_ub, bound_method, hl, wl, hu, wu):
    N = net.call_model("get_num")
    gs_batch = net.call_model("get_gs_batch")
    bg_color=(net.call_model("get_bg_color_tile")).unsqueeze(0).unsqueeze(-2) #[1, TH, TW, N, 3]

    if N==0:
        return bg_color.squeeze(-2), bg_color.squeeze(-2)
    else:
        alphas_int_lb = []
        alphas_int_ub = []

        hl,wl,hu,wu = (net.call_model("get_tile_dict")[key] for key in ["hl", "wl", "hu", "wu"])

        # Combine input bounds with tile bounds (lower with lower, upper with upper)
        tile_lb = torch.tensor([[hl, wl]], device=input_lb.device, dtype=input_lb.dtype)
        tile_ub = torch.tensor([[hu, wu]], device=input_ub.device, dtype=input_ub.dtype)
        combined_lb = torch.cat([input_lb, tile_lb], dim=-1)
        combined_ub = torch.cat([input_ub, tile_ub], dim=-1)
        combined_ref = torch.cat([input_ref, 0.5 * (tile_lb + tile_ub)], dim=-1)

        ptb = PerturbationLpNorm(x_L=combined_lb, x_U=combined_ub)
        input_ptb = BoundedTensor(combined_ref, ptb)

        with torch.no_grad():
            for i, idx_start in enumerate(range(0, N, gs_batch)):
                idx_end = min(idx_start + gs_batch, N)

                net.call_model("update_model_param",idx_start,idx_end,"middle")
                model = BoundedModule(net, combined_ref, bound_opts=bound_opts, device=DEVICE)

                # Compute IBP bounds for reference
                alpha_ibp_lb, alpha_ibp_ub = model.compute_bounds(x=(input_ptb, ), method="ibp")
                reference_interm_bounds = {}
                for node in model.nodes():
                    if (node.perturbed
                        and isinstance(node.lower, torch.Tensor)
                        and isinstance(node.upper, torch.Tensor)):
                        reference_interm_bounds[node.name] = (node.lower, node.upper)

                # required_A = defaultdict(set)
                # required_A[model.output_name[0]].add(model.input_name[0])

                # Compute linear buond for alpha
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

        # Volume Rendering for Interval Bounds
        color_alpha_out_lb, color_alpha_out_ub = alpha_blending_interval(alphas_int_lb, alphas_int_ub, colors)

        color_out_lb,alpha_out_lb = color_alpha_out_lb.split([3,1],dim=-1)
        color_out_ub,alpha_out_ub = color_alpha_out_ub.split([3,1],dim=-1)

    return color_out_lb.squeeze(-2), color_out_ub.squeeze(-2)
