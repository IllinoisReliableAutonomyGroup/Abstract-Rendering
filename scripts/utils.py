import os

import numpy as np
import torch
import json

from scipy.spatial.transform import Rotation 
import itertools

def transfer_c2w_to_w2c(c2w):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = c2w[:, :3, :3]  # 3 x 3
    T = c2w[:, :3, 3:4]  # 3 x 1

    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)

    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)

    w2c = torch.stack([
        torch.cat([R_inv[:, 0], T_inv[:, 0]], dim=1),  # first row
        torch.cat([R_inv[:, 1], T_inv[:, 1]], dim=1),  # second row
        torch.cat([R_inv[:, 2], T_inv[:, 2]], dim=1),  # third row
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=R.device, dtype=R.dtype).expand(R.shape[0], -1)  # last row
    ], dim=1)  # (B, 4, 4)
    return w2c

def quaternion_to_rotation_matrix(quats):
    """
    Converts quaternions to rotation matrices.
    Args:
        quats: Tensor of shape [N, 4], where each quaternion is [w, x, y, z].
    Returns:
        Rotation matrices of shape [N, 3, 3].
    """
    quats = quats / quats.norm(dim=1, keepdim=True)
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    N = quats.size(0)
    R = torch.empty(N, 3, 3, device=quats.device, dtype=quats.dtype)

    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)

    return R 

# def dir_to_rpy_and_rot(target, pose):

#     direction = target-pose

#     direction = np.array(direction, dtype=float)
#     direction = direction / np.linalg.norm(direction)

#     yaw = np.arctan2(direction[1], direction[0])
#     pitch = -np.arcsin(direction[2])

#     rotation_matrix = Rotation.from_euler('xyz',[0,pitch,yaw]).as_matrix()
#     R = Rotation.from_euler('yzx',[np.pi/2, 0, -np.pi/2]).as_matrix()
#     rot_matrix = rotation_matrix@R

#     R = Rotation.from_euler('yzx',[0, -np.pi/2, 0]).as_matrix()
#     rot_matrix = rot_matrix@R

#     return rot_matrix

def dir_to_rpy_and_rot(target, pose):
    forward = target - pose
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0, 1, 0], dtype=float)
    # If forward is parallel to world_up, choose another up vector
    if np.allclose(np.abs(np.dot(forward, world_up)), 1.0):
        world_up = np.array([0, 0, 1], dtype=float)

    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    rot_matrix = np.column_stack((right, up, forward))  # columns: right, up, forward
    return rot_matrix


def convert_input_to_translation(input, trans, type):
        input = input.to(trans.device)

        if type == "3":
            r, theta_v, theta_h = input[:, 0], input[:, 1], input[:, 2]
            
            y = r * torch.sin(theta_v)
            xz = r * torch.cos(theta_v)
            z = xz * torch.sin(theta_h)
            x = -xz * torch.cos(theta_h)
        elif type == "x":

            x = input[:, 0]+trans[0]
            y = torch.ones_like(x)*trans[1]
            z = torch.ones_like(x)*trans[2]
        elif type == "y":

            y = input[:, 0]+trans[1]
            x = torch.ones_like(y)*trans[0]
            z = torch.ones_like(y)*trans[2]
        elif type == "z":

            z = input[:, 0]
            x = torch.ones_like(z)*trans[0]
            y = torch.ones_like(z)*trans[1]
        elif type == "yaw":

            yaw = input[:, 0]

            y = torch.ones_like(yaw)*trans[1]
            xz = trans[0]
            z = -xz * torch.sin(yaw)
            x = xz * torch.cos(yaw)

        elif type == "round":
            yaw  = input[:, 0]
            initial_yaw = torch.arctan2(-trans[2],trans[0])

            y  = torch.ones_like(yaw)*trans[1]
            xz = torch.sqrt(trans[0]**2+trans[2]**2)
            z = -xz * torch.sin(yaw+initial_yaw)
            x = xz * torch.cos(yaw+initial_yaw)

        return torch.stack([x, y, z], dim=-1)

def convert_input_to_rot(input, trans, type):
    translation = convert_input_to_translation(input, trans, type)
    translation = translation.detach().cpu().numpy().squeeze(0) #[3, ]

    center = np.zeros((3,))

    rot = dir_to_rpy_and_rot(translation, center)

    return rot

def convert_input_to_pose(input, rot, trans, transform_hom, scale, type):
    
    DEVICE = rot.device
    DTYPE = rot.dtype
    translation = convert_input_to_translation(input, trans, type)  # [B, 3]
    translation = translation.to(DEVICE)

    #print(translation.device, rot.device, input.device)
    #print(translation.dtype, rot.dtype, input.dtype)
    # print(rot.shape, input.shape, translation.shape)
    # Build transformation matrix in shape (B, 4, 4)
    B = translation.shape[0]
    trans_mat = torch.cat([
        torch.cat([rot.unsqueeze(0).expand(B, -1, -1), translation.unsqueeze(-1)], dim=-1),  # (B, 3, 4)
        torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=DTYPE, device=DEVICE).view(1, 1, 4).expand(B, -1, -1)  # (b, 1, 4)
    ], dim=1)  # (B, 4, 4)

    # print("trans_mat:", trans_mat)

    tmp = torch.eye(4, dtype = DTYPE, device=DEVICE)

    #print(tmp.dtype, transform_hom.dtype, trans_mat.dtype)
    c2w = tmp @ transform_hom @ trans_mat
    scaled_translation = c2w[:, :3, 3] * scale  # [B, 3]
    c2w = torch.cat([c2w[:, :3, :3], scaled_translation.unsqueeze(-1)], dim=-1)  # [B, 3, 4]

    # Convert to view matrix
    w2c = transfer_c2w_to_w2c(c2w)

    return w2c 

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

def alpha_blending_interval_2(colors_alpha_lb, colors_alpha_ub):

    colors_lb,alpha_lb = colors_alpha_lb.split([3,1],dim=-1) 
    colors_ub,alpha_ub = colors_alpha_ub.split([3,1],dim=-1)
    
    alpha_lb_shifted = torch.cat([torch.zeros_like(alpha_lb[:,:,:,0:1,:], dtype=alpha_lb.dtype), alpha_lb[:,:,:,:-1,:]], dim=-2)
    transmittance_ub = regulate(torch.cumprod((1-alpha_lb_shifted), dim=-2))

    alpha_ub_shifted = torch.cat([torch.zeros_like(alpha_ub[:,:,:,0:1,:], dtype=alpha_lb.dtype), alpha_ub[:,:,:,:-1,:]], dim=-2)
    transmittance_lb = regulate(torch.cumprod((1-alpha_ub_shifted), dim=-2))

    alpha_out_lb = regulate(torch.sum((alpha_lb*transmittance_lb), dim=-2, keepdim=True))
    alpha_out_ub = regulate(torch.sum((alpha_ub*transmittance_ub), dim=-2, keepdim=True))

    color_out_lb = regulate(torch.sum((alpha_lb*transmittance_lb*colors_lb), dim=-2, keepdim=True))
    color_out_ub = regulate(torch.sum((alpha_ub*transmittance_ub*colors_ub), dim=-2, keepdim=True))

    color_alpha_out_lb = torch.cat([color_out_lb,alpha_out_lb], dim = -1)
    color_alpha_out_ub = torch.cat([color_out_ub,alpha_out_ub], dim = -1)
    return color_alpha_out_lb, color_alpha_out_ub

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

def generate_single(input_lb, input_ub, input_ref, N_sample=5):
    assert torch.all(input_lb <= input_ub), "input_lb must be <= input_ub"

    N = input_lb.shape[1]  # number of dimensions

    return input_ref

def generate_samples(input_lb, input_ub, input_ref, N_sample=5, include_given=True):
    assert torch.all(input_lb <= input_ub), "input_lb must be <= input_ub"

    N = input_lb.shape[1]  # number of dimensions
    
    # Generate uniform random samples within bounds
    rand_vals = torch.rand((N_sample, N), device=input_lb.device)
    samples = input_lb + (input_ub - input_lb) * rand_vals

    # Concatenate with bounds
    if include_given:
        input_samples = torch.cat([input_ref, input_lb, input_ub, samples], dim=0) #(N_sample+3, N)
    else:
        input_samples = torch.cat([samples], dim=0) #(N_sample, N)

    return input_samples


def generate_trajectory(input_lb, input_ub, input_ref, N_sample=20, N_waypoints=3):
    """
    Generate a trajectory (smooth-ish path) within bounds.
    - input_lb: (1, d) lower bound
    - input_ub: (1, d) upper bound
    - input_ref: (1, d) reference point
    - N_sample: number of points in trajectory
    - N_waypoints: number of random waypoints (in addition to start & end)
    """
    d = input_lb.shape[1]

    # Random start and end inside bounds
    start = input_lb + (input_ub - input_lb) * torch.rand((1, d), device=input_lb.device)
    end   = input_lb + (input_ub - input_lb) * torch.rand((1, d), device=input_lb.device)

    # Random intermediate waypoints
    waypoints = [start]
    for _ in range(N_waypoints):
        wp = input_lb + (input_ub - input_lb) * torch.rand((1, d), device=input_lb.device)
        waypoints.append(wp)
    waypoints.append(end)

    # Concatenate waypoints
    waypoints = torch.cat(waypoints, dim=0)  # (N_waypoints+2, d)

    # Allocate trajectory
    traj = []

    # Divide samples approximately evenly across each segment
    seg_lengths = torch.norm(waypoints[1:] - waypoints[:-1], dim=1)
    seg_ratios = seg_lengths / seg_lengths.sum()
    seg_counts = (seg_ratios * N_sample).long()
    seg_counts[seg_counts == 0] = 1  # at least one per segment

    # Build trajectory by interpolating segments
    for i in range(len(waypoints)-1):
        n = seg_counts[i].item()
        alpha = torch.linspace(0, 1, steps=n, device=input_lb.device).unsqueeze(1)
        seg_points = (1-alpha) * waypoints[i] + alpha * waypoints[i+1]
        traj.append(seg_points)

    traj = torch.cat(traj, dim=0)

    # Concatenate with reference point
    input_samples = torch.cat([input_ref, traj], dim=0)

    return input_samples


def generate_bound(input_min, input_max, partition_per_dim=2, selection_per_dim=None):
    if selection_per_dim is None:
        selection_per_dim = partition_per_dim

    N = input_min.size(0)

    # Generate boundaries for each dimension
    grids = []
    idx_choices = []
    for i in range(N):
        edges = torch.linspace(input_min[i], input_max[i], steps=partition_per_dim+1 )
        grids.append(edges)

        if selection_per_dim >= partition_per_dim:
            choices = list(range(partition_per_dim))  # use all intervals
        else:
            choices = torch.linspace(
                0, partition_per_dim-1 , steps=selection_per_dim, dtype=torch.long
            ).tolist()
        idx_choices.append(choices)

    inputs_lb = []
    inputs_ub = []
    inputs_ref = []

    # Cartesian product of partitions in each dimension
    for indices in itertools.product(*idx_choices, repeat=N):
        lb = torch.tensor([grids[dim][idx] for dim, idx in enumerate(indices)])
        ub = torch.tensor([grids[dim][idx + 1] for dim, idx in enumerate(indices)])
        ref = (lb + ub) / 2

        inputs_lb.append(lb)
        inputs_ub.append(ub)
        inputs_ref.append(ref)

    # Stack into tensors
    inputs_lb = torch.stack(inputs_lb)
    inputs_ub = torch.stack(inputs_ub)
    inputs_ref = torch.stack(inputs_ref)

    return inputs_lb, inputs_ub, inputs_ref

if __name__ == '__main__':
    x=torch.rand(1,1,1,5,1)
    triu_mask = torch.triu(torch.ones(7, 7), diagonal=1)
    print(x)
    y = cumprod(x, triu_mask)
    print(y)