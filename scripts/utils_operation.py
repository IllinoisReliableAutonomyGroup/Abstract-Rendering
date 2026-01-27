import numpy as np
import torch

def regulate(x, eps_min=0.0, eps_max=1.0):
    x = -torch.nn.functional.relu(-x+eps_max)+eps_max 
    x = torch.nn.functional.relu(x-eps_min)+eps_min
    return x

def cumsum(x, triu_mask, dim=-2):
    N = x.size(-2)
    triu_mask = triu_mask[:N, :N].to(x.device)
    cumsum_x = (triu_mask[None, None, None, :, :] * x).sum(dim=dim, keepdim=True) # [1, TH, TW, 1, B]
    cumsum_x = cumsum_x.transpose(-1,-2) # [1, TH, TW, B, 1]

    return cumsum_x

def cumprod(x, triu_mask, dim=-2, eps_min=1e-8):
    x = torch.nn.functional.relu(x-eps_min)+eps_min 
    return torch.exp(cumsum(torch.log(x), triu_mask, dim=dim))

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


if __name__ == '__main__':
    pass