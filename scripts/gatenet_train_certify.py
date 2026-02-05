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
    def __init__(self, samples_json, image_root):
        with open(samples_json, 'r') as f:
            self.samples = json.load(f)
        self.image_root = image_root
        
        print(f"Loaded {len(self.samples)} samples from {samples_json}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Construct image filename: ref_000000.png, ref_000001.png, etc.
        img_filename = f"ref_{sample['index']:06d}.png"
        img_path = os.path.join(self.image_root, img_filename)
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
        
        # Get relative pose (this is X_center - the 3D position)
        pose = torch.tensor(sample['relative_pose'], dtype=torch.float32)  # [x, y, z]
        
        return img, pose


class AbstractImageDataset(Dataset):
    """Dataset for abstract images with bound information"""
    def __init__(self, abstract_folder):
        self.abstract_folder = abstract_folder
        self.pt_files = sorted([f for f in os.listdir(abstract_folder) if f.endswith('.pt')])
        
        print(f"Found {len(self.pt_files)} abstract .pt files in {abstract_folder}")
        
    def __len__(self):
        return len(self.pt_files)
    
    def __getitem__(self, idx):
        pt_path = os.path.join(self.abstract_folder, self.pt_files[idx])
        data = torch.load(pt_path, weights_only=False)
        
        # Extract data
        lower_img = data['lower']  # [H, W, 3]
        upper_img = data['upper']  # [H, W, 3]
        X_lower = data['xl']  # [3]
        X_upper = data['xu']  # [3]
        
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


def abstract_loss(Y_lower, Y_upper, X_lower, X_upper):
    """MSE loss for abstract images: MSE(Y_lower, X_lower) + MSE(Y_upper, X_upper)"""
    loss_lower = nn.MSELoss()(Y_lower, X_lower)
    loss_upper = nn.MSELoss()(Y_upper, X_upper)
    return loss_lower + loss_upper


# ============================================================================
# CROWN Bound Propagation
# ============================================================================

def crown_propagate(model, lower_img, upper_img, bound_method='forward'):
    """
    Propagate bounds through GateNet using auto_LiRPA
    
    Args:
        model: GateNet model
        lower_img: [B, 3, H, W] lower bound images
        upper_img: [B, 3, H, W] upper bound images
        bound_method: 'backward', 'forward', 'crown-ibp', or 'ibp'
    
    Returns:
        Y_lower: [B, 3] lower bounds on output
        Y_upper: [B, 3] upper bounds on output
    """
    # Use center as reference input
    img_center = (lower_img + upper_img) / 2.0
    
    # Wrap model with BoundedModule
    bound_opts = {
        'conv_mode': 'matrix',
    }
    lirpa_model = BoundedModule(model, img_center, bound_opts=bound_opts, device=DEVICE)
    
    # Define perturbation using explicit bounds
    ptb = PerturbationLpNorm(x_L=lower_img, x_U=upper_img)
    img_ptb = BoundedTensor(img_center, ptb)
    
    # Compute bounds

    Y_lower, Y_upper = lirpa_model.compute_bounds(x=(img_ptb,), method=bound_method)
    
    del lirpa_model
    del img_ptb
    torch.cuda.empty_cache()
    
    return Y_lower, Y_upper


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, concrete_loader, abstract_loader, optimizer, config, epoch):
    """Train for one epoch"""
    model.train()
    
    lambda_concrete = config['lambda_concrete']
    lambda_abstract = config['lambda_abstract']
    bound_method = config['bound_method']
    
    # Create iterators
    concrete_iter = iter(concrete_loader)
    abstract_iter = iter(abstract_loader)
    
    total_loss = 0.0
    total_concrete_loss = 0.0
    total_abstract_loss = 0.0
    num_steps = 0
    
    # Determine number of iterations (use the longer loader)
    num_iters = max(len(concrete_loader), len(abstract_loader))
    
    pbar = tqdm(range(num_iters), desc=f"Epoch {epoch}")
    
    for step in pbar:
        optimizer.zero_grad()
        
        loss_c = 0.0
        loss_a = 0.0
        
        # ===== Concrete Loss =====
        try:
            concrete_imgs, concrete_poses = next(concrete_iter)
            concrete_imgs = concrete_imgs.to(DEVICE)
            concrete_poses = concrete_poses.to(DEVICE)
            
            # Forward pass
            predictions = model(concrete_imgs)
            loss_c = concrete_loss(predictions, concrete_poses)
            
        except StopIteration:
            # Restart iterator if exhausted
            concrete_iter = iter(concrete_loader)
            loss_c = 0.0
        
        # ===== Abstract Loss =====
        try:
            lower_imgs, upper_imgs, X_lowers, X_uppers = next(abstract_iter)
            lower_imgs = lower_imgs.to(DEVICE)
            upper_imgs = upper_imgs.to(DEVICE)
            X_lowers = X_lowers.to(DEVICE)
            X_uppers = X_uppers.to(DEVICE)
            
            # Propagate bounds through model
            Y_lowers, Y_uppers = crown_propagate(model, lower_imgs, upper_imgs, bound_method)
            
            # Compute abstract loss (averaged over batch)
            loss_a = abstract_loss(Y_lowers, Y_uppers, X_lowers, X_uppers)
            
        except StopIteration:
            # Restart iterator if exhausted
            abstract_iter = iter(abstract_loader)
            loss_a = 0.0
        
        # ===== Combined Loss =====
        loss_total = lambda_concrete * loss_c + lambda_abstract * loss_a
        
        # Backward and optimize
        if loss_total > 0:
            loss_total.backward()
            optimizer.step()
        
        # Logging
        total_loss += loss_total.item()
        total_concrete_loss += loss_c.item() if isinstance(loss_c, torch.Tensor) else 0.0
        total_abstract_loss += loss_a.item() if isinstance(loss_a, torch.Tensor) else 0.0
        num_steps += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_total.item():.4f}',
            'loss_c': f'{loss_c.item() if isinstance(loss_c, torch.Tensor) else 0.0:.4f}',
            'loss_a': f'{loss_a.item() if isinstance(loss_a, torch.Tensor) else 0.0:.4f}'
        })
    
    # Return average losses
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
        image_root=config['concrete_image_root']
    )
    abstract_dataset = AbstractImageDataset(
        abstract_folder=config['abstract_folder']
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
    
    # ===== Training Loop =====
    print("\n=== Starting Training ===")
    best_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        losses = train_epoch(model, concrete_loader, abstract_loader, optimizer, config, epoch)
        
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print(f"  Total Loss: {losses['total']:.6f}")
        print(f"  Concrete Loss: {losses['concrete']:.6f}")
        print(f"  Abstract Loss: {losses['abstract']:.6f}")
        
        # Save checkpoint every N epochs
        if epoch % config['save_every'] == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses['total'],
                'config': config
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if losses['total'] < best_loss:
            best_loss = losses['total']
            best_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, best_path)
            print(f"  ✓ New best model saved: {best_path} (loss={best_loss:.6f})")
    
    # Save final model
    final_path = os.path.join(config['checkpoint_dir'], 'final_model.pth')
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses['total'],
        'config': config
    }, final_path)
    print(f"\n=== Training Complete ===")
    print(f"Final model saved: {final_path}")
    print(f"Best model (loss={best_loss:.6f}): {best_path}")


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