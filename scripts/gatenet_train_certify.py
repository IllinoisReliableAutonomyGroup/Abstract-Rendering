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
    def __init__(self, abstract_folder, target_size=(64, 64)):
        self.abstract_folder = abstract_folder
        self.target_size = target_size

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
        
        if lower_img.shape[0] != self.target_size[1] or lower_img.shape[1] != self.target_size[0]:
            lower_img = F.interpolate(
                lower_img.permute(2, 0, 1).unsqueeze(0),
                size=(self.target_size[1], self.target_size[0]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)
            
            upper_img = F.interpolate(
                upper_img.permute(2, 0, 1).unsqueeze(0),
                size=(self.target_size[1], self.target_size[0]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)

        # Convert images to [3, H, W] format
        lower_img = lower_img.permute(2, 0, 1)  # [3, H, W]
        upper_img = upper_img.permute(2, 0, 1)  # [3, H, W]
        
        return lower_img, upper_img, X_lower, X_upper


# ============================================================================
# Loss Functions
# ============================================================================

def concrete_loss(predictions, targets):
    """MSE loss for concrete images"""
    return nn.MSELoss()(predictions, targets)


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
    Propagate bounds through GateNet using auto_LiRPA with gradients
    """
    img_center = (lower_img + upper_img) / 2.0
    
    bound_opts = {'conv_mode': 'patches'}
    lirpa_model = BoundedModule(model, img_center, bound_opts=bound_opts, device=DEVICE)
    
    ptb = PerturbationLpNorm(x_L=lower_img, x_U=upper_img)
    img_ptb = BoundedTensor(img_center, ptb)
    
    # NO torch.no_grad() - forward method supports gradients!
    Y_lower, Y_upper = lirpa_model.compute_bounds(x=(img_ptb,), method=bound_method)
    return Y_lower, Y_upper

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
    use_reinforce = config.get('use_reinforce', False)  # New config option
    noise_std = config.get('reinforce_noise_std', 0.1)  # New config option
    
    concrete_iter = iter(concrete_loader)
    abstract_iter = iter(abstract_loader)
    
    total_loss = 0.0
    total_concrete_loss = 0.0
    total_abstract_loss = 0.0
    num_steps = 0
    
    num_iters = max(len(concrete_loader), len(abstract_loader))
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
            lower_imgs, upper_imgs, X_lowers, X_uppers = next(abstract_iter)
            lower_imgs = lower_imgs.to(DEVICE)
            upper_imgs = upper_imgs.to(DEVICE)
            X_lowers = X_lowers.to(DEVICE)
            X_uppers = X_uppers.to(DEVICE)
            
            optimizer.zero_grad()
            
            if use_reinforce:
                # REINFORCE method: gradients through Y_center only
                reinforce_loss, loss_a_val = reinforce_abstract_gradient(
                    model, lower_imgs, upper_imgs, X_lowers, X_uppers, 
                    bound_method, noise_std
                )
                (lambda_abstract * reinforce_loss).backward()
            else:
                # Direct method: gradients through CROWN bounds
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
        method_str = "REINFORCE" if use_reinforce else "Direct"
        pbar.set_postfix({
            'loss': f'{loss_total_val:.4f}',
            'loss_c': f'{loss_c_val:.4f}',
            'loss_a': f'{loss_a_val:.4f}',
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
    concrete_dataset = ConcreteImageDataset(
        samples_json=config['concrete_samples_json'],
        image_root=config['concrete_image_root'],
        target_size=(config['image_width'], config['image_height'])
    )
    abstract_dataset = AbstractImageDataset(
        abstract_folder=config['abstract_folder'],
        target_size=(config['image_width'], config['image_height'])
    )
    
    print(f"Concrete samples: {len(concrete_dataset)}")
    print(f"Abstract samples: {len(abstract_dataset)}")
    
    # ===== Create DataLoaders =====
    concrete_loader = DataLoader(
        concrete_dataset,
        batch_size=config['batch_size_concrete'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
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

    # ===== Prepare run directory (timestamped) =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_ckpt_dir = config['checkpoint_dir']
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

    # If best was never updated (should not happen), fall back to final state
    if best_state_dict is None:
        best_state_dict = model.state_dict()
        best_epoch = config['num_epochs']

    # Save best model (latest) at the end
    best_path = os.path.join(run_dir, 'latest.pth')
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
        'config': config
    }, best_path)

    # Save final model (last epoch) at the end
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

    # ===== Plot loss curves over epochs =====
    epochs = list(range(1, config['num_epochs'] + 1))
    plt.figure()
    eps = 1e-8
    plt.plot(epochs, [max(v, eps) for v in total_history], label='total loss')
    plt.plot(epochs, [max(v, eps) for v in concrete_history], label='concrete loss')
    plt.plot(epochs, [max(v, eps) for v in abstract_history], label='abstract loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.title('Training losses (log scale)')
    plt.legend()

    plot_filename = f"{config['bound_method']}_lambda{config['lambda_concrete']}.png"
    plot_path = os.path.join(run_dir, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved: {plot_path}")


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