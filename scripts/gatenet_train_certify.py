import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import gc

grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
if grandfather_path not in sys.path:
    sys.path.append(grandfather_path)

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from DownStreamModel.gatenet.gatenet import GateNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Dataset Classes
# ============================================================================

class ConcreteImageDataset(Dataset):
    """Dataset for concrete (real) images with pose labels"""
    def __init__(self, samples_json, image_root, target_size=(64, 64)):
        with open(samples_json, 'r') as f:
            all_samples = json.load(f)
        self.image_root = image_root

        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

        skipped = sum(1 for s in all_samples if not s.get('include', True))
        self.samples = [s for s in all_samples if s.get('include', True)]
        print(f"Loaded {len(self.samples)} samples from {samples_json} ({skipped} skipped — include=False)")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Construct image filename: ref_000000.png, ref_000001.png, etc.
        img_filename = f"ref_{sample['index']:06d}.png"
        img_path = os.path.join(self.image_root, img_filename)
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        # Get relative pose (this is X_center - the 3D position)
        pose = torch.tensor(sample['relative_pose'], dtype=torch.float32)  # [x, y, z]
        
        return img, pose


class AbstractImageDataset(Dataset):
    """Dataset for abstract images with bound information"""
    def __init__(self, abstract_folder, target_size=(64, 64), use_lsr=False):
        self.abstract_folder = abstract_folder
        self.target_size = target_size
        self.use_lsr = use_lsr

        all_pt_files = sorted([f for f in os.listdir(abstract_folder) if f.endswith('.pt')])

        # Only keep files that have valid lower_rel/upper_rel (GT relative pose bounds)
        self.pt_files = []
        skipped = 0
        for fname in all_pt_files:
            data = torch.load(os.path.join(abstract_folder, fname), weights_only=False)
            if data.get('lower_rel') is not None and data.get('upper_rel') is not None:
                self.pt_files.append(fname)
            else:
                skipped += 1

        print(f"Found {len(self.pt_files)} abstract .pt files in {abstract_folder} ({skipped} skipped — no lower_rel/upper_rel)")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = os.path.join(self.abstract_folder, self.pt_files[idx])
        data = torch.load(pt_path, weights_only=False)

        # Extract data
        lower_img = data['lower']  # [H, W, 3]
        upper_img = data['upper']  # [H, W, 3]
        X_lower = data['lower_rel']  # [3] GT cuboid lower bound on relative pose
        X_upper = data['upper_rel']  # [3] GT cuboid upper bound on relative pose

        need_resize = (lower_img.shape[0] != self.target_size[1] or
                       lower_img.shape[1] != self.target_size[0])

        if self.use_lsr:
            # LSR mode: pixel A matrices are tied to image spatial layout — no resize allowed
            if need_resize:
                raise ValueError(
                    f"use_lsr=True but record {self.pt_files[idx]} has size "
                    f"{lower_img.shape[:2]} vs target {self.target_size}. "
                    "Set target_size to match the record resolution."
                )
            lA_px = data['lA']   # [H, W, 3, n_x]
            uA_px = data['uA']
            lb_px = data['lb']   # [H, W, 3]
            ub_px = data['ub']
            xl    = data['xl']   # [n_x]
            xu    = data['xu']

            lower_img = lower_img.permute(2, 0, 1)  # [3, H, W]
            upper_img = upper_img.permute(2, 0, 1)
            return lower_img, upper_img, X_lower, X_upper, lA_px, uA_px, lb_px, ub_px, xl, xu

        if need_resize:
            lower_img = F.interpolate(
                lower_img.permute(2, 0, 1).unsqueeze(0),
                size=(self.target_size[1], self.target_size[0]),
                mode='bilinear', align_corners=False
            ).squeeze(0).permute(1, 2, 0)
            upper_img = F.interpolate(
                upper_img.permute(2, 0, 1).unsqueeze(0),
                size=(self.target_size[1], self.target_size[0]),
                mode='bilinear', align_corners=False
            ).squeeze(0).permute(1, 2, 0)

        lower_img = lower_img.permute(2, 0, 1)  # [3, H, W]
        upper_img = upper_img.permute(2, 0, 1)
        return lower_img, upper_img, X_lower, X_upper


# ============================================================================
# Loss Functions
# ============================================================================

def concrete_loss(predictions, targets):
    """MSE loss for concrete images"""
    return nn.MSELoss()(predictions, targets)


def lsr_loss(pose_lA, pose_lb_bias, pose_uA, pose_ub_bias, xl, xu, X_lower, X_upper):
    """
    Soundness loss for LSR training.

    Certified worst-case bounds over x ∈ [xl, xu]:
        cert_worst_upper[i] = relu(pose_uA[i]) @ xu - relu(-pose_uA[i]) @ xl + pose_ub_bias[i]
        cert_worst_lower[i] = relu(pose_lA[i]) @ xl - relu(-pose_lA[i]) @ xu + pose_lb_bias[i]

    Loss penalises violations where the certified range does not contain [X_lower, X_upper].

    Args:
        pose_lA, pose_uA : [B, n_out, n_x]
        pose_lb_bias, pose_ub_bias : [B, n_out]
        xl, xu           : [B, n_x]   per-sample perturbation bounds
        X_lower, X_upper : [B, n_out] GT relative pose bounds
    """
    # xu/xl: [B, n_x] → [B, 1, n_x] to broadcast over n_out
    xl_ = xl.unsqueeze(1)
    xu_ = xu.unsqueeze(1)

    # Width of certified error interval at worst-case x ∈ [xl, xu]:
    # gap(x) = (pose_uA - pose_lA)·x + (pose_ub - pose_lb)
    # max over x: relu(diff_A) @ xu - relu(-diff_A) @ xl + (pose_ub - pose_lb)
    diff_A    = pose_uA - pose_lA                                    # [B, n_out, n_x]
    worst_gap = (torch.relu( diff_A) * xu_ -
                 torch.relu(-diff_A) * xl_).sum(dim=-1) \
              + (pose_ub_bias - pose_lb_bias)                        # [B, n_out]

    return worst_gap.mean()


def abstract_loss(Y_lower, Y_upper, X_lower, X_upper, tolerance=0.02):
    """
    Soundness-preserving loss: X_lower ≤ Y_lower ≤ Y_upper ≤ X_upper
    tolerance: allowed slack before penalizing violations
    """
    violation_lower = torch.relu(X_lower - Y_lower - tolerance).mean()

    # Penalize if Y_upper > X_upper (predicted upper bound should be conservative)
    violation_upper = torch.relu(Y_upper - X_upper - tolerance).mean()

    # Penalize if Y_lower > Y_upper (invalid interval)
    bound_validity = torch.relu(Y_lower - Y_upper).mean()

    return violation_lower + violation_upper + 5.0 * bound_validity


# ============================================================================
# CROWN Bound Propagation
# ============================================================================

def crown_propagate(model, lower_img, upper_img, bound_method='forward'):
    """
    Propagate bounds through GateNet using auto_LiRPA with gradients.
    Uses patches conv_mode (tight, convolutional structure preserved) for training.
    """
    img_center = (lower_img + upper_img) / 2.0

    bound_opts = {'conv_mode': 'patches'}
    lirpa_model = BoundedModule(model, img_center, bound_opts=bound_opts, device=DEVICE)

    ptb = PerturbationLpNorm(x_L=lower_img, x_U=upper_img)
    img_ptb = BoundedTensor(img_center, ptb)

    Y_lower, Y_upper = lirpa_model.compute_bounds(x=(img_ptb,), method=bound_method)
    return Y_lower, Y_upper


_lsr_debug_calls = 0  # module-level counter: print debug only for first 3 calls

def crown_propagate_lsr(model, lower_imgs, upper_imgs, lA_px, uA_px, lb_px, ub_px):
    """
    LSR-aware bound propagation for training — single batched CROWN pass.

    1. Run CROWN (conv_mode='matrix', return_A=True) on the full batch
    2. Compose per-sample A matrices with the pixel LSR via batched matmul
    3. Return pose A matrices used in lsr_loss

    Args:
        lower_imgs, upper_imgs : [B, 3, H, W]
        lA_px, uA_px           : [B, H, W, 3, n_x]
        lb_px, ub_px           : [B, H, W, 3]

    Returns:
        pose_lA, pose_uA           : [B, n_out, n_x]
        pose_lb_bias, pose_ub_bias : [B, n_out]
    """
    B       = lower_imgs.shape[0]
    n_x     = lA_px.shape[4]
    n_pixels = lA_px.shape[1] * lA_px.shape[2] * lA_px.shape[3]  # C*H*W

    img_center = 0.5 * (lower_imgs + upper_imgs)

    ptb     = PerturbationLpNorm(x_L=lower_imgs, x_U=upper_imgs)
    img_ptb = BoundedTensor(img_center, ptb)

    # conv_mode='matrix': full dense A at input — required for composition with lA_px
    lirpa_model = BoundedModule(model, img_center,
                                bound_opts={"conv_mode": "matrix"}, device=DEVICE)

    # Specify which A matrices to return: output → input
    needed_A = {lirpa_model.output_name[0]: [lirpa_model.input_name[0]]}

    # No torch.no_grad() — gradients must flow through A matrices to update weights
    out_lb, out_ub, A_dict = lirpa_model.compute_bounds(
        x=(img_ptb,), method="backward", return_A=True, needed_A_dict=needed_A
    )

    # Access: A_dict[output_name][input_name] → {lA, uA, lbias, ubias}
    A_entry = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]
    # [B, n_out, C, H, W] → [B, n_out, n_pixels]
    lA_net    = A_entry["lA"].reshape(B, -1, n_pixels)
    uA_net    = A_entry["uA"].reshape(B, -1, n_pixels)
    lbias_net = A_entry["lbias"]   # [B, n_out] — constant term in CROWN linear bound
    ubias_net = A_entry["ubias"]
    n_out  = lA_net.shape[1]

    # Pixel LSR: [B, H, W, 3, n_x] → [B, 3, H, W, n_x] → [B, n_pixels, n_x]
    lA_r = lA_px.permute(0, 3, 1, 2, 4).reshape(B, n_pixels, n_x)
    uA_r = uA_px.permute(0, 3, 1, 2, 4).reshape(B, n_pixels, n_x)
    lb_r = lb_px.permute(0, 3, 1, 2).reshape(B, n_pixels)     # [B, n_pixels]
    ub_r = ub_px.permute(0, 3, 1, 2).reshape(B, n_pixels)

    # Lower bound composition — batched matmul via bmm
    lA_pos = torch.relu( lA_net)   # [B, n_out, n_pixels]
    lA_neg = torch.relu(-lA_net)
    pose_lA      = torch.bmm(lA_pos, lA_r) - torch.bmm(lA_neg, uA_r)          # [B, n_out, n_x]
    pose_lb_bias = (lA_pos * lb_r.unsqueeze(1)).sum(-1) \
                 - (lA_neg * ub_r.unsqueeze(1)).sum(-1) + lbias_net             # [B, n_out]

    # Upper bound composition
    uA_pos = torch.relu( uA_net)
    uA_neg = torch.relu(-uA_net)
    pose_uA      = torch.bmm(uA_pos, uA_r) - torch.bmm(uA_neg, lA_r)          # [B, n_out, n_x]
    pose_ub_bias = (uA_pos * ub_r.unsqueeze(1)).sum(-1) \
                 - (uA_neg * lb_r.unsqueeze(1)).sum(-1) + ubias_net             # [B, n_out]

    global _lsr_debug_calls
    if _lsr_debug_calls < 1:
        _lsr_debug_calls += 1
        tqdm.write(f"[DEBUG] lA_net requires_grad={lA_net.requires_grad}, grad_fn={lA_net.grad_fn}")
        tqdm.write(f"[DEBUG] lbias_net requires_grad={lbias_net.requires_grad}, grad_fn={lbias_net.grad_fn}")
        tqdm.write(f"[DEBUG] pose_lA requires_grad={pose_lA.requires_grad}, grad_fn={pose_lA.grad_fn}")
        tqdm.write(f"[DEBUG] pose_lb_bias requires_grad={pose_lb_bias.requires_grad}, grad_fn={pose_lb_bias.grad_fn}")

    return pose_lA, pose_lb_bias, pose_uA, pose_ub_bias


def make_lirpa_model(model, image_shape, device):
    """
    Create a BoundedModule for GateNet once and reuse across all certification calls.
    image_shape: (C, H, W)
    """
    dummy = torch.zeros(1, *image_shape, device=device)
    lirpa_model = BoundedModule(model, dummy,
                                bound_opts={"conv_mode": "matrix"}, device=device)
    return lirpa_model


def crown_certify_lsr(lirpa_model, record, device):
    """
    Certify a single abstract record with Linear Set Representation.

    Composes CROWN's per-pixel A matrices with the pixel LSR stored in the
    record to produce affine pose bounds w.r.t. the original perturbation x:

        lA_pose · x + lb_pose  ≤  pose  ≤  uA_pose · x + ub_pose,   x ∈ [-1,1]^n_x

    lirpa_model: pre-created BoundedModule (reused across all records to save memory).

    Returns:
        out_lb      [1, n_out]   interval lower bound on pose
        out_ub      [1, n_out]   interval upper bound on pose
        pose_lA     [n_out, n_x] or None if record has no LSR
        pose_lb_bias[n_out]      or None
        pose_uA     [n_out, n_x] or None
        pose_ub_bias[n_out]      or None
    """
    lower_img = record["lower"].to(device)   # [H, W, 3]
    upper_img = record["upper"].to(device)
    lA_px = record.get("lA")                 # [H, W, 3, n_x] or None
    uA_px = record.get("uA")
    lb_px = record.get("lb")                 # [H, W, 3] or None
    ub_px = record.get("ub")

    # Reorder to [1, 3, H, W] for GateNet
    lower_r = lower_img.permute(2, 0, 1).unsqueeze(0)
    upper_r = upper_img.permute(2, 0, 1).unsqueeze(0)
    img_center = 0.5 * (lower_r + upper_r)

    ptb = PerturbationLpNorm(x_L=lower_r, x_U=upper_r)
    img_ptb = BoundedTensor(img_center, ptb)

    needed_A = {lirpa_model.output_name[0]: [lirpa_model.input_name[0]]}
    out_lb, out_ub, A_dict = lirpa_model.compute_bounds(
        x=(img_ptb,), method="backward", return_A=True, needed_A_dict=needed_A
    )

    # ── Compose network A matrices with pixel LSR ─────────────────────────────
    pose_lA = pose_lb_bias = pose_uA = pose_ub_bias = None
    if lA_px is not None:
        # A_dict[output_name][input_name] → {lA, uA, lbias, ubias}
        A_entry  = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]
        lA_net   = A_entry["lA"].squeeze(0)    # [n_out, C, H, W]
        uA_net   = A_entry["uA"].squeeze(0)
        lbias_net = A_entry["lbias"].squeeze(0) # [n_out]
        ubias_net = A_entry["ubias"].squeeze(0)
        n_out    = lA_net.shape[0]
        n_pixels = lA_net[0].numel()
        n_x      = lA_px.shape[3]
        lA_net = lA_net.reshape(n_out, n_pixels)
        uA_net = uA_net.reshape(n_out, n_pixels)

        # [H, W, 3, n_x] → [3, H, W, n_x] → [C*H*W, n_x]  (matches CHW pixel order)
        def _px(t):
            return t.permute(2, 0, 1, 3).reshape(n_pixels, n_x).to(device)
        def _bias(t):
            return t.permute(2, 0, 1).reshape(n_pixels).to(device)

        lA_r = _px(lA_px);   uA_r = _px(uA_px)
        lb_r = _bias(lb_px); ub_r = _bias(ub_px)

        # Lower bound: relu-split on lA_net
        lA_pos = torch.relu( lA_net)
        lA_neg = torch.relu(-lA_net)
        pose_lA      = lA_pos @ lA_r - lA_neg @ uA_r              # [n_out, n_x]
        pose_lb_bias = lA_pos @ lb_r  - lA_neg @ ub_r + lbias_net # [n_out]

        # Upper bound: relu-split on uA_net
        uA_pos = torch.relu( uA_net)
        uA_neg = torch.relu(-uA_net)
        pose_uA      = uA_pos @ uA_r - uA_neg @ lA_r              # [n_out, n_x]
        pose_ub_bias = uA_pos @ ub_r  - uA_neg @ lb_r + ubias_net # [n_out]

    return out_lb, out_ub, pose_lA, pose_lb_bias, pose_uA, pose_ub_bias

# ============================================================================
# REINFORCE Gradient Estimation
# ============================================================================

def reinforce_abstract_gradient(model, lower_imgs, upper_imgs, X_lower, X_upper, bound_method='backward', noise_std=0.1):
    """
    REINFORCE-style gradient estimation for abstract loss
    Enables training with abstract bounds without backpropping through CROWN
    """
    img_center = (lower_imgs + upper_imgs) / 2.0
    
    # Get CROWN bounds in eval mode (detached)
    with torch.no_grad():
        model.eval()
        Y_L_crown, Y_U_crown = crown_propagate(model, lower_imgs, upper_imgs, bound_method)
        loss_a = abstract_loss(Y_L_crown, Y_U_crown, X_lower, X_upper)
        reward = -loss_a.item()
        model.train()
    
    # Get model's prediction on center image (HAS gradients!)
    Y_center = model(img_center)
    
    # Compute REINFORCE loss with detached CROWN bounds
    diff_L = (Y_L_crown.detach() - Y_center) / noise_std
    diff_U = (Y_U_crown.detach() - Y_center) / noise_std
    
    log_prob = -0.5 * ((diff_L ** 2).sum() + (diff_U ** 2).sum())
    reinforce_loss = -reward * log_prob
    
    return reinforce_loss, loss_a.item()

# ============================================================================
# Training Loop
# ============================================================================

# def train_epoch(model, concrete_loader, abstract_loader, optimizer, config, epoch):
#     """Train for one epoch with BOTH concrete and abstract losses"""
#     model.train()
    
#     lambda_concrete = config['lambda_concrete']
#     lambda_abstract = config['lambda_abstract']
#     bound_method = config['bound_method']
    
#     concrete_iter = iter(concrete_loader)
#     abstract_iter = iter(abstract_loader)
    
#     total_loss = 0.0
#     total_concrete_loss = 0.0
#     total_abstract_loss = 0.0
#     num_steps = 0
    
#     num_iters = max(len(concrete_loader), len(abstract_loader))
#     pbar = tqdm(range(num_iters), desc=f"Epoch {epoch}")
    
#     for step in pbar:
#         loss_c_val = 0.0
#         loss_a_val = 0.0
        
#         # ===== Concrete Loss (separate step) =====
#         try:
#             concrete_imgs, concrete_poses = next(concrete_iter)
#             concrete_imgs = concrete_imgs.to(DEVICE)
#             concrete_poses = concrete_poses.to(DEVICE)
            
#             optimizer.zero_grad()
#             predictions = model(concrete_imgs)
#             loss_c = concrete_loss(predictions, concrete_poses)
#             (lambda_concrete * loss_c).backward()
#             optimizer.step()
            
#             loss_c_val = loss_c.item()
            
#         except StopIteration:
#             concrete_iter = iter(concrete_loader)
#             loss_c_val = 0.0
        
#         # ===== Abstract Loss (separate step) =====
#         try:
#             lower_imgs, upper_imgs, X_lowers, X_uppers = next(abstract_iter)
#             lower_imgs = lower_imgs.to(DEVICE)
#             upper_imgs = upper_imgs.to(DEVICE)
#             X_lowers = X_lowers.to(DEVICE)
#             X_uppers = X_uppers.to(DEVICE)
            
#             optimizer.zero_grad()
            
#             # Get bounds from CROWN propagation (has gradients!)
#             Y_lower_pred, Y_upper_pred = crown_propagate(model, lower_imgs, upper_imgs, bound_method)
            
#             # Loss: predictions should match ground truth bounds
#             loss_a = abstract_loss(Y_lower_pred, Y_upper_pred, X_lowers, X_uppers)
#             (lambda_abstract * loss_a).backward()
#             optimizer.step()
            
#             loss_a_val = loss_a.item()
            
#         except StopIteration:
#             abstract_iter = iter(abstract_loader)
#             loss_a_val = 0.0
        
#         # ===== Logging =====
#         loss_total_val = lambda_concrete * loss_c_val + lambda_abstract * loss_a_val
        
#         total_loss += loss_total_val
#         total_concrete_loss += loss_c_val
#         total_abstract_loss += loss_a_val
#         num_steps += 1
        
#         # Update progress bar
#         pbar.set_postfix({
#             'loss': f'{loss_total_val:.4f}',
#             'loss_c': f'{loss_c_val:.4f}',
#             'loss_a': f'{loss_a_val:.4f}'
#         })
    
#     return {
#         'total': total_loss / num_steps,
#         'concrete': total_concrete_loss / num_steps,
#         'abstract': total_abstract_loss / num_steps
#     }

def train_epoch(model, concrete_loader, abstract_loader, optimizer, config, epoch):
    """Train for one epoch with BOTH concrete and abstract losses"""
    model.train()
    
    lambda_concrete = config['lambda_concrete']
    lambda_abstract = config['lambda_abstract']
    bound_method = config['bound_method']
    tolerance = config.get('tolerance', 0.02)
    use_reinforce = config.get('use_reinforce', False)
    noise_std = config.get('reinforce_noise_std', 0.1)
    use_lsr = config.get('use_lsr', False)
    
    concrete_iter = iter(concrete_loader)
    abstract_iter = iter(abstract_loader)
    
    total_loss = 0.0
    total_concrete_loss = 0.0
    total_abstract_loss = 0.0
    num_steps = 0

    num_iters = max(len(concrete_loader), len(abstract_loader)) if concrete_loader else len(abstract_loader)
    pbar = tqdm(range(num_iters), desc=f"Epoch {epoch}")
    
    for step in pbar:
        loss_c_val = 0.0
        loss_a_val = 0.0
        
        # ===== Concrete Loss (separate step) =====
        try:
            concrete_imgs, concrete_poses = next(concrete_iter)
            concrete_imgs = concrete_imgs.to(DEVICE)
            concrete_poses = concrete_poses.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(concrete_imgs)
            loss_c = concrete_loss(predictions, concrete_poses)
            (lambda_concrete * loss_c).backward()
            optimizer.step()
            
            loss_c_val = loss_c.item()
            
        except StopIteration:
            concrete_iter = iter(concrete_loader)
            loss_c_val = 0.0
        
        # ===== Abstract Loss (separate step) =====
        try:
            abstract_batch = next(abstract_iter)
            lower_imgs = abstract_batch[0].to(DEVICE)
            upper_imgs = abstract_batch[1].to(DEVICE)
            X_lowers   = abstract_batch[2].to(DEVICE)
            X_uppers   = abstract_batch[3].to(DEVICE)

            optimizer.zero_grad()

            if use_lsr:
                # auto_LiRPA detaches A matrices (requires_grad=False) — cannot
                # backprop through them.  Train with the differentiable interval
                # bounds (out_lb / out_ub) over the pixel interval [lower, upper],
                # which is valid for all x ∈ [xl, xu].
                # A-matrix composition into the 3×3 pose LSR is done at certification.
                Y_lower_pred, Y_upper_pred = crown_propagate(model, lower_imgs, upper_imgs, bound_method)
                # Tightness loss: minimize certified interval width.
                # Always > 0 since CROWN overapproximates ReLU activations.
                loss_a = (Y_upper_pred - Y_lower_pred).mean()
                (lambda_abstract * loss_a).backward()
                loss_a_val = loss_a.item()
            elif use_reinforce:
                # REINFORCE method: gradients through Y_center only
                reinforce_loss, loss_a_val = reinforce_abstract_gradient(
                    model, lower_imgs, upper_imgs, X_lowers, X_uppers,
                    bound_method, noise_std
                )
                (lambda_abstract * reinforce_loss).backward()
            else:
                # Direct method: gradients through CROWN interval bounds
                Y_lower_pred, Y_upper_pred = crown_propagate(model, lower_imgs, upper_imgs, bound_method)
                loss_a = abstract_loss(Y_lower_pred, Y_upper_pred, X_lowers, X_uppers, tolerance=tolerance)
                (lambda_abstract * loss_a).backward()
                loss_a_val = loss_a.item()

            optimizer.step()
            
        except StopIteration:
            abstract_iter = iter(abstract_loader)
            loss_a_val = 0.0
        
        # ===== Logging =====
        loss_total_val = lambda_concrete * loss_c_val + lambda_abstract * loss_a_val
        
        total_loss += loss_total_val
        total_concrete_loss += loss_c_val
        total_abstract_loss += loss_a_val
        num_steps += 1
        
        # Update progress bar
        method_str = "REINFORCE" if use_reinforce else ("LSR" if use_lsr else "Direct")
        pbar.set_postfix({
            'loss': f'{loss_total_val:.6f}',
            'loss_c': f'{loss_c_val:.6f}',
            'loss_a': f'{loss_a_val:.6f}',
            'method': method_str
        })
    
    return {
        'total': total_loss / num_steps,
        'concrete': total_concrete_loss / num_steps,
        'abstract': total_abstract_loss / num_steps
    }


# ============================================================================
# Main Training Function
# ============================================================================

def main(config):
    print(f"Device: {DEVICE}")
    print(f"\n=== Configuration ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # ===== Create Datasets =====
    print("\n=== Loading Datasets ===")
    abstract_dataset = AbstractImageDataset(
        abstract_folder=config['abstract_folder'],
        target_size=(config['image_width'], config['image_height']),
        use_lsr=config.get('use_lsr', False)
    )
    print(f"Abstract samples: {len(abstract_dataset)}")

    use_concrete = config.get('lambda_concrete', 0.0) > 0.0
    if use_concrete:
        concrete_dataset = ConcreteImageDataset(
            samples_json=config['concrete_samples_json'],
            image_root=config['concrete_image_root'],
            target_size=(config['image_width'], config['image_height'])
        )
        print(f"Concrete samples: {len(concrete_dataset)}")
    else:
        concrete_dataset = None
        print("Concrete dataset skipped (lambda_concrete=0)")

    # ===== Create DataLoaders =====
    if use_concrete:
        concrete_loader = DataLoader(
            concrete_dataset,
            batch_size=config['batch_size_concrete'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
    else:
        concrete_loader = []  # empty — train_epoch will skip concrete step
    abstract_loader = DataLoader(
        abstract_dataset,
        batch_size=config['batch_size_abstract'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Concrete batches per epoch: {len(concrete_loader)}")
    print(f"Abstract batches per epoch: {len(abstract_loader)}")
    print(f"Training iterations per epoch: {max(len(concrete_loader), len(abstract_loader))}")
    
    # ===== Create Model =====
    print("\n=== Creating Model ===")
    model_config = {
        'input_shape': (3, config['image_height'], config['image_width']),
        'output_shape': (3,),  # x, y, z
        'batch_norm_decay': config['batch_norm_decay'],
        'batch_norm_epsilon': config['batch_norm_epsilon']
    }
    model = GateNet(model_config).to(DEVICE)

    # Load pretrained checkpoint if specified
    if config.get('pretrained_checkpoint') is not None:
        checkpoint_path = config['pretrained_checkpoint']
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded pretrained weights from: {checkpoint_path}")
            if 'epoch' in checkpoint:
                print(f"  Checkpoint was from epoch {checkpoint['epoch']}")
        else:
            print(f"⚠ Warning: Checkpoint not found at {checkpoint_path}")
            print(f"  Training from scratch instead.")
    else:
        print("Training from scratch (no pretrained checkpoint)")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ===== Create Optimizer =====
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # ===== Prepare run directory =====
    base_ckpt_dir = config['checkpoint_dir']
    if config['num_epochs'] == 0 and config.get('pretrained_checkpoint'):
        # Certify-only: reuse the directory that holds the pretrained checkpoint
        run_dir = os.path.dirname(os.path.abspath(config['pretrained_checkpoint']))
        timestamp = os.path.basename(run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_ckpt_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nCheckpoints, plots, and settings will be saved under: {run_dir}")

    # Save a text summary of the run settings
    settings_path = os.path.join(run_dir, "run_settings.txt")
    with open(settings_path, "w") as f:
        f.write("GateNet train+certify run settings\n")
        f.write(f"timestamp: {timestamp}\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    print(f"Run settings saved to: {settings_path}")

    # ===== Training Loop =====
    print("\n=== Starting Training ===")
    best_loss = float('inf')
    best_state_dict = None
    best_epoch = None
    total_history = []
    concrete_history = []
    abstract_history = []

    for epoch in range(1, config['num_epochs'] + 1):
        losses = train_epoch(model, concrete_loader, abstract_loader, optimizer, config, epoch)
        
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print(f"  Total Loss: {losses['total']:.6f}")
        print(f"  Concrete Loss: {losses['concrete']:.6f}")
        print(f"  Abstract Loss: {losses['abstract']:.6f}")

        # Record losses for plotting
        total_history.append(losses['total'])
        concrete_history.append(losses['concrete'])
        abstract_history.append(losses['abstract'])

        # Track best model (by total loss) but do not save until the end
        if losses['total'] < best_loss:
            best_loss = losses['total']
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    # If num_epochs=0 (certify-only), use the pretrained model weights directly
    if best_state_dict is None:
        best_state_dict = model.state_dict()
        best_epoch = 0

    if config['num_epochs'] > 0:
        # Save best model
        best_path = os.path.join(run_dir, 'latest.pth')
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'config': config
        }, best_path)

        # Save final model
        final_path = os.path.join(run_dir, 'final_model.pth')
        torch.save({
            'epoch': config['num_epochs'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses['total'],
            'config': config
        }, final_path)
        print(f"\n=== Training Complete ===")
        print(f"Final model saved: {final_path}")
        print(f"Best model (epoch={best_epoch}, loss={best_loss:.6f}): {best_path}")

        # Plot loss curves
        epochs_list = list(range(1, config['num_epochs'] + 1))
        plt.figure()
        eps = 1e-8
        plt.plot(epochs_list, [max(v, eps) for v in total_history],    label='total loss')
        plt.plot(epochs_list, [max(v, eps) for v in concrete_history], label='concrete loss')
        plt.plot(epochs_list, [max(v, eps) for v in abstract_history], label='abstract loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss (log scale)'); plt.yscale('log')
        plt.title('Training losses (log scale)'); plt.legend()
        plot_filename = f"{config['bound_method']}_lambda{config['lambda_concrete']}.png"
        plot_path = os.path.join(run_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight'); plt.close()
        print(f"Loss curves saved: {plot_path}")
    else:
        best_path = config['pretrained_checkpoint']
        print(f"\n=== Skipped Training (num_epochs=0) — using {best_path} ===")

    # ===== LSR Certification Pass =====
    # Load the best model and run crown_certify_lsr over the abstract folder.
    # Produces affine pose bounds w.r.t. the original cuboid perturbation x.
    print("\n=== LSR Certification ===")
    best_model = GateNet(model_config).to(DEVICE)
    best_ckpt = torch.load(best_path)
    best_model.load_state_dict(best_ckpt["model_state_dict"])
    best_model.eval()

    abstract_folder = config["abstract_folder"]
    all_pt_files = sorted([f for f in os.listdir(abstract_folder) if f.startswith("abstract_") and f.endswith(".pt")])

    cert_out_lb_all      = []
    cert_out_ub_all      = []
    cert_pose_lA_all     = []
    cert_pose_lb_bias_all= []
    cert_pose_uA_all     = []
    cert_pose_ub_bias_all= []
    has_lsr = True

    # Certify on GPU — BoundedModule created ONCE and reused; clear cache each step
    cert_device = DEVICE
    image_shape = (3, config['image_height'], config['image_width'])
    lirpa_cert = make_lirpa_model(best_model, image_shape, cert_device)

    cert_path = os.path.join(run_dir, "lsr_certification.pt")
    save_every = 50  # save partial results every N partitions

    # Resume from existing checkpoint if present
    resume_from = 0
    if os.path.exists(cert_path):
        existing = torch.load(cert_path, map_location='cpu', weights_only=False)
        resume_from = existing["out_lb"].shape[0]
        cert_out_lb_all      = list(existing["out_lb"])
        cert_out_ub_all      = list(existing["out_ub"])
        if "pose_lA" in existing:
            cert_pose_lA_all      = list(existing["pose_lA"])
            cert_pose_lb_bias_all = list(existing["pose_lb_bias"])
            cert_pose_uA_all      = list(existing["pose_uA"])
            cert_pose_ub_bias_all = list(existing["pose_ub_bias"])
        print(f"  Resuming certification from partition {resume_from}/{len(all_pt_files)}")

    for i, fname in enumerate(all_pt_files):
        if i < resume_from:
            continue
        rec = torch.load(os.path.join(abstract_folder, fname), weights_only=False)
        out_lb, out_ub, pose_lA, pose_lb_bias, pose_uA, pose_ub_bias = \
            crown_certify_lsr(lirpa_cert, rec, cert_device)
        cert_out_lb_all.append(out_lb.squeeze(0).cpu())
        cert_out_ub_all.append(out_ub.squeeze(0).cpu())
        if pose_lA is not None:
            cert_pose_lA_all.append(pose_lA.cpu())
            cert_pose_lb_bias_all.append(pose_lb_bias.cpu())
            cert_pose_uA_all.append(pose_uA.cpu())
            cert_pose_ub_bias_all.append(pose_ub_bias.cpu())
        else:
            has_lsr = False

        # Free GPU memory each step
        gc.collect()
        if cert_device.type == 'cuda':
            torch.cuda.empty_cache()

        if (i + 1) % save_every == 0 or (i + 1) == len(all_pt_files):
            cert_results = {
                "out_lb": torch.stack(cert_out_lb_all,  dim=0),
                "out_ub": torch.stack(cert_out_ub_all,  dim=0),
            }
            if has_lsr and cert_pose_lA_all:
                cert_results["pose_lA"]      = torch.stack(cert_pose_lA_all,       dim=0)
                cert_results["pose_lb_bias"] = torch.stack(cert_pose_lb_bias_all,  dim=0)
                cert_results["pose_uA"]      = torch.stack(cert_pose_uA_all,       dim=0)
                cert_results["pose_ub_bias"] = torch.stack(cert_pose_ub_bias_all,  dim=0)
            torch.save(cert_results, cert_path)
            print(f"  [{i+1}/{len(all_pt_files)}] checkpoint saved → {cert_path}")

            # Recreate BoundedModule to flush accumulated intermediate bound caches
            if (i + 1) < len(all_pt_files):
                del lirpa_cert
                gc.collect()
                if cert_device.type == 'cuda':
                    torch.cuda.empty_cache()
                lirpa_cert = make_lirpa_model(best_model, image_shape, cert_device)

    print(f"LSR certification complete: {cert_path}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train GateNet with Concrete + Abstract Images")
    parser.add_argument('--config', type=str, required=True, help='Path to training config YAML')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Run training
    main(config)