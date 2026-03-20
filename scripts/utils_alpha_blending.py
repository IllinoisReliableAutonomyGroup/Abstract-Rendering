import os
import sys
import torch
import torch.nn as nn

from utils_operation import regulate, cumprod

grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(grandfather_path)

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto_LiRPA.perturbations import PerturbationLinear
from collections import defaultdict



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bound_opts = {
    'conv_mode': 'matrix',
    'optimize_bound_args': {
        'iteration': 100,
        # 'lr_alpha':0.02,
        'early_stop_patience':5},
}


class SumCumProdModel(nn.Module):
    """SUMCUMPROD (Algorithm 5) as nn.Module for auto_LiRPA.

    Input:  alpha, shape (B, N)
    Output: pc, shape (B, 3) — all 3 color channels at once
    Colors are fixed (not perturbed).
    """
    def __init__(self, colors):
        super().__init__()
        self.register_buffer('colors', colors)  # (B, N, 3)

    def forward(self, alpha):
        B, N = alpha.shape
        log_one_minus_alpha = torch.log(torch.relu(1 - alpha - 1e-6) + 1e-6)   # (B, N), safe: clamps 1-alpha >= 1e-6

        running = torch.zeros(B, 1, device=alpha.device)
        prefix_list = []
        for i in range(N):
            prefix_list.append(running)
            running = running + log_one_minus_alpha[:, i:i+1]

        prefix = torch.cat(prefix_list, dim=1)        # (B, N)
        prefix = -torch.relu(-prefix)                  # clamp ≤ 0
        prefix = torch.exp(prefix)                     # (B, N)

        alpha = torch.relu(alpha)
        color = torch.relu(self.colors)
        return torch.sum(prefix.unsqueeze(-1) * alpha.unsqueeze(-1) * color, dim=1)

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


def alpha_blending_ptb(net, input_ref, input_lb, input_ub, bound_method):
    N = net.call_model("get_num")
    gs_batch = net.call_model("get_gs_batch")
    bg_color = net.call_model("get_bg_color_tile").unsqueeze(0).unsqueeze(-2)  # (1, TH, TW, 1, 3)

    if N == 0:
        return bg_color.squeeze(-2), bg_color.squeeze(-2)

    # ── STEP 1: Extract linear bounds of α (per GS batch) from auto_LiRPA ──
    alphas_lA = []
    alphas_uA = []
    alphas_lbias = []
    alphas_ubias = []
    alphas_ref = []

    hl, wl, hu, wu = (net.call_model("get_tile_dict")[key]
                       for key in ["hl", "wl", "hu", "wu"])
    TH, TW = hu - hl, wu - wl

    ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
    input_ptb = BoundedTensor(input_ref, ptb)

    with torch.no_grad():
        for idx_start in range(0, N, gs_batch):
            idx_end = min(idx_start + gs_batch, N)
            num_gs = idx_end - idx_start

            net.call_model("update_model_param", idx_start, idx_end, "middle")

            # Forward pass at input_ref for exact reference alphas
            alpha_ref_batch = net(input_ref)   # (1, TH*TW*num_gs)
            alpha_ref_batch = alpha_ref_batch.reshape(TH * TW, num_gs)
            alphas_ref.append(alpha_ref_batch.detach())

            model = BoundedModule(net, input_ref, bound_opts=bound_opts, device=DEVICE)

            # IBP for reference intermediate bounds
            model.compute_bounds(x=(input_ptb,), method="ibp")
            reference_interm_bounds = {}
            for node in model.nodes():
                if (node.perturbed
                        and isinstance(node.lower, torch.Tensor)
                        and isinstance(node.upper, torch.Tensor)):
                    reference_interm_bounds[node.name] = (node.lower, node.upper)

            # CROWN with A matrix extraction
            required_A = defaultdict(set)
            required_A[model.output_name[0]].add(model.input_name[0])

            # Must use 'backward' (CROWN) to extract A matrices;
            # forward mode does not support return_A.
            _, _, A_dict = model.compute_bounds(
                x=(input_ptb,),
                method='backward',
                reference_bounds=reference_interm_bounds,
                return_A=True,
                needed_A_dict=required_A,
            )

            # lA·x + lbias ≤ α ≤ uA·x + ubias
            A_entry = A_dict[model.output_name[0]][model.input_name[0]]
            lA    = A_entry['lA'].detach()      # (1, TH*TW*num_gs, input_dim)
            uA    = A_entry['uA'].detach()
            lbias = A_entry['lbias'].detach()    # (1, TH*TW*num_gs)
            ubias = A_entry['ubias'].detach()

            # Reshape per-pixel: (1, TH*TW*num_gs, d) → (TH*TW, num_gs, d)
            lA    = lA.reshape(TH * TW, num_gs, -1)
            uA    = uA.reshape(TH * TW, num_gs, -1)
            lbias = lbias.reshape(TH * TW, num_gs)
            ubias = ubias.reshape(TH * TW, num_gs)

            alphas_lA.append(lA)
            alphas_uA.append(uA)
            alphas_lbias.append(lbias)
            alphas_ubias.append(ubias)

        del model
        torch.cuda.empty_cache()

    # ── STEP 2: Concatenate α (partial GS) → α (all GS) ──
    # Concatenate along the Gaussian dimension (dim=1 for per-pixel A matrices)
    full_lA    = torch.cat(alphas_lA, dim=1)       # (TH*TW, N, input_dim)
    full_uA    = torch.cat(alphas_uA, dim=1)
    full_lbias = torch.cat(alphas_lbias, dim=1)    # (TH*TW, N)
    full_ubias = torch.cat(alphas_ubias, dim=1)
    alpha_ref_all = torch.cat(alphas_ref, dim=1)   # (TH*TW, N)

    # Append background Gaussian: α_bg = 1 (fixed, zero A, bias = 1)
    num_pixels = TH * TW
    input_dim = full_lA.shape[-1]
    full_lA    = torch.cat([full_lA,
                            torch.zeros(num_pixels, 1, input_dim, device=DEVICE)], dim=1)
    full_uA    = torch.cat([full_uA,
                            torch.zeros(num_pixels, 1, input_dim, device=DEVICE)], dim=1)
    full_lbias = torch.cat([full_lbias,
                            torch.ones(num_pixels, 1, device=DEVICE)], dim=1)
    full_ubias = torch.cat([full_ubias,
                            torch.ones(num_pixels, 1, device=DEVICE)], dim=1)
    alpha_ref_flat = torch.cat([alpha_ref_all,
                                torch.ones(num_pixels, 1, device=DEVICE)], dim=1)  # (TH*TW, N+1)

    # Colors + background
    colors = net.call_model("get_color_tile")
    colors = colors.view(1, 1, 1, N, 3).repeat(1, TH, TW, 1, 1)
    colors = torch.cat([colors, bg_color], dim=-2)   # (1, TH, TW, N+1, 3)

    # ── STEPS 3–5: PerturbationLinear → BoundedModule → compute_bounds ──
    N_total = full_lA.shape[1]   # N + 1 (including background)

    # Colors: (1, TH, TW, N_total, 3) → (TH*TW, N_total, 3)
    colors_flat = colors.reshape(-1, N_total, 3)

    with torch.no_grad():
        # STEP 3: Wrap α bounds in PerturbationLinear
        # PerturbationLinear.concretize() will compose A_blend (from CROWN on
        # blending model) with lA/uA and concretize against input_lb/input_ub,
        # preserving full correlation: camera x → α → pc.
        ptb_linear = PerturbationLinear(
            lower_A=full_lA,                                       # (TH*TW, N_total, input_dim)
            upper_A=full_uA,
            lower_b=full_lbias,                                    # (TH*TW, N_total)
            upper_b=full_ubias,
            input_lb=input_lb.expand(num_pixels, -1),              # (TH*TW, input_dim)
            input_ub=input_ub.expand(num_pixels, -1),
        )
        alpha_ptb = BoundedTensor(alpha_ref_flat, ptb_linear)

        # STEP 4: Wrap blending step in BoundedModule (all 3 channels at once)
        blend_model = SumCumProdModel(colors_flat)
        bounded_blend = BoundedModule(
            blend_model, alpha_ref_flat,
            bound_opts=bound_opts, device=DEVICE
        )

        # STEP 5: CROWN backward → A_blend; PerturbationLinear.concretize
        # composes A_blend with lA/uA and concretizes against camera bounds.
        pixel_lb, pixel_ub = bounded_blend.compute_bounds(
            x=(alpha_ptb,), method="backward"
        )
        # pixel_lb, pixel_ub: (TH*TW, 3)

        img_lb = pixel_lb.reshape(1, TH, TW, 3)
        img_ub = pixel_ub.reshape(1, TH, TW, 3)

        del bounded_blend
        torch.cuda.empty_cache()

    # ── STEP 6: Return image bounds ──
    return img_lb, img_ub