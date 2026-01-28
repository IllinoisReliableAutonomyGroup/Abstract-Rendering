import os

import numpy as np
import torch
import json

from scipy.spatial.transform import Rotation 


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

def pose_to_matrix(trans, rot, transform_hom, scale):
    rot = rot.to(device=trans.device, dtype=trans.dtype)
    transform_hom = transform_hom.to(device=trans.device, dtype = trans.dtype)
    scale = scale.to(device=trans.device, dtype = trans.dtype)

    tmp = Rotation.from_euler('zyx',[-np.pi/2,np.pi/2,0]).as_matrix()
    tmp = torch.from_numpy(tmp).to(device=trans.device, dtype=trans.dtype)
    rot_corrected = rot@tmp
    

    B = trans.shape[0]
    viewmats = torch.cat([
        torch.cat([rot_corrected.unsqueeze(0).expand(B, -1, -1), trans.unsqueeze(-1)], dim=-1),  # (B, 3, 4)
        torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=trans.dtype, device=trans.device).view(1, 1, 4).expand(B, -1, -1)  # (b, 1, 4)
    ], dim=1)  # (B, 4, 4)
    

    # viewmats[:, 0:3, 1:3] *= -1
    # viewmats = viewmats[:, torch.tensor([0, 2, 1, 3], device=trans.device), :]
    # viewmats[:, 2, :] *= -1

    # flip YZ columns
    viewmats = torch.cat([viewmats[:, :, :1], -viewmats[:, :, 1:3], viewmats[:, :, 3:]], dim=2)

    # swap rows 1 ↔ 2
    viewmats = torch.cat([viewmats[:, :1], viewmats[:, 2:3], viewmats[:, 1:2], viewmats[:, 3:]],dim=1)

    # flip row 2
    viewmats = torch.cat([viewmats[:, :2], -viewmats[:, 2:3], viewmats[:, 3:]],dim=1)

    
    viewmats = transform_hom @ viewmats
    scaled_translation = viewmats[:, :3, 3] * scale  # [B, 3]
    viewmats = torch.cat([viewmats[:, :3, :3], scaled_translation.unsqueeze(-1)], dim=-1)  # [B, 3, 4]

    # Convert to view matrix
    viewmats = transfer_c2w_to_w2c(viewmats)

    return viewmats

def orthogonal_basis_from_direction(direction, eps=1e-8):
    """
    direction: torch.Tensor of shape (3,)
    returns: direction, orthogonal1, orthogonal2 (all shape (3,))
    """
    device = direction.device
    dtype = direction.dtype

    # Normalize direction
    direction = direction / (torch.norm(direction) + eps)

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

    # Check if direction is (approximately) ±x-axis
    is_x_axis = torch.allclose(direction, x_axis, atol=1e-6) or \
                torch.allclose(direction, -x_axis, atol=1e-6)

    if is_x_axis:
        orthogonal1 = y_axis
    else:
        orthogonal1 = torch.linalg.cross(direction, x_axis)
        orthogonal1 = orthogonal1 / (torch.norm(orthogonal1) + eps)

    orthogonal2 = torch.linalg.cross(direction, orthogonal1)
    return direction, orthogonal1, orthogonal2

def input_to_trans(input, base_trans, type, direction=None, radius=None):
    base_trans = base_trans.to(input.device)

    if type == "cylinder":
        dir_coeff, rad_coeff, angle = input[:, 0], input[:, 1], input[:, 2]

        if direction is None:
            raise ValueError("direction vector must be provided for 'cylinder' type.")
        if radius is None:
            raise ValueError("radius must be provided for 'cylinder' type.")
        else:
            direction, radius = direction.to(input.device), radius.to(input.device)
            t, o1, o2 = orthogonal_basis_from_direction(direction)
            point = (
                dir_coeff*direction.unsqueeze(0) +
                rad_coeff*radius * torch.cos(angle).unsqueeze(1) * o1.unsqueeze(0) +
                rad_coeff*radius * torch.sin(angle).unsqueeze(1) * o2.unsqueeze(0)
            )
            point+=base_trans.unsqueeze(0)

    return point


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32
    type = "cylinder"
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_folder = os.path.join(script_dir, '.')

    samples_file = os.path.join(script_dir, '../configs/traj.json')
    with open(samples_file, 'r') as f:
        samples = json.load(f)

    print(f"len(samples): {len(samples)}")
    sample = samples[0]
    sample_next = samples[1]

    pose = sample["pose"]
    pose_next = sample_next["pose"]
    trans, orientation = pose[:3], pose[3:]
    direction = np.array(pose_next[:3]) - np.array(pose[:3])
    direction = torch.tensor(direction, device=DEVICE, dtype=DTYPE)
    radius = sample["radius"]

    trans = torch.tensor(trans, device=DEVICE, dtype=DTYPE)

    R = Rotation.from_euler('ZYX', orientation)
    rot = R.as_matrix()
    rot = torch.tensor(rot, device=DEVICE, dtype=DTYPE)

    print(f"trans, or orientation: {trans, orientation}")

    ### Inspect transform from gsplat scene
    gsplat_path = os.path.join(script_dir, '../nerfstudio/outputs/uturn/splatfacto/2025-05-09_151825')
    transform_file = f"{gsplat_path}/dataparser_transforms.json"

    with open (transform_file, 'r') as fp:
        data_transform = json.load(fp)
        transform = np.array (data_transform['transform'])
        transform_hom = np.vstack((transform, np.array([0,0,0,1])))
        scale = data_transform['scale']

    transform_hom = torch.from_numpy(transform_hom).to(dtype=DTYPE, device=DEVICE)
    scale = torch.tensor(scale,dtype=DTYPE, device=DEVICE)

    viewmats= pose_to_matrix(trans.unsqueeze(0), rot, transform_hom, scale)
    print(f"viewmats : {viewmats}")

    ### Inspect input_to_trans
    
    input = [1.0,1.0,0.0]
    input = torch.tensor(input, device=DEVICE, dtype=DTYPE).unsqueeze(0)
    
    d, o1, o2 = orthogonal_basis_from_direction(direction)
    print(f"d, o1, o2: {d, o1, o2}")

    print(f"dirction, d:{direction, d}")

    point = input_to_trans(input, trans, type, direction, radius)
    print(f"point: {point}")
