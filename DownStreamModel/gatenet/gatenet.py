import torch
import torch.nn as nn
import torch.nn.functional as F

class GateNet(nn.Module):
    def __init__(self, config):
        super(GateNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16, momentum=config['batch_norm_decay'], eps=config['batch_norm_epsilon'])
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32, momentum=config['batch_norm_decay'], eps=config['batch_norm_epsilon'])

        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(16, momentum=config['batch_norm_decay'], eps=config['batch_norm_epsilon'])

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(16, momentum=config['batch_norm_decay'], eps=config['batch_norm_epsilon'])

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.bn5 = nn.BatchNorm2d(16, momentum=config['batch_norm_decay'], eps=config['batch_norm_epsilon'])

        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.bn6 = nn.BatchNorm2d(16, momentum=config['batch_norm_decay'], eps=config['batch_norm_epsilon'])

        self.flatten = nn.Flatten()

        # print(config['input_shape'])
        res = self.conv(torch.zeros(config['input_shape'])[None])
        # print(res.shape[1])
        self.fc = nn.Linear(res.shape[1], int(torch.prod(torch.tensor(config['output_shape']))))

    def conv(self, x):
        x = F.relu(self.bn1(self.conv1(x))) # 64
        x = F.avg_pool2d(x, kernel_size=2)

        x = F.relu(self.bn2(self.conv2(x))) # 32
        x = F.avg_pool2d(x, kernel_size=2)

        x = F.relu(self.bn3(self.conv3(x))) # 16
        x = F.avg_pool2d(x, kernel_size=2)

        x = F.relu(self.bn4(self.conv4(x))) # 8
        x = F.avg_pool2d(x, kernel_size=2)

        x = F.relu(self.bn5(self.conv5(x))) # 4
        x = F.avg_pool2d(x, kernel_size=2)


        x = F.relu(self.bn6(self.conv6(x)))  # No pooling after conv6  # 2

        x = self.flatten(x)

        return x

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    import os
    from PIL import Image 
    import numpy as np
    import argparse
    from pathlib import Path 
    import re 
    from typing import Optional

    def latest_checkpoint(folder: str) -> Optional[Path]:
        """
        Return the checkpoint file with the highest epoch index in *folder*.
        
        Parameters
        ----------
        folder : str | Path
            Directory containing files named 'checkpoint_epoch_{i}.pth'.
        
        Returns
        -------
        pathlib.Path | None
            Path to the newest checkpoint, or None if none are found.
        """
        folder_path = Path(folder)
        pattern = re.compile(r"^checkpoint_epoch_(\d+)\.pth$")

        candidates = []
        for fp in folder_path.iterdir():
            if fp.is_file():
                m = pattern.match(fp.name)
                if m:
                    epoch = int(m.group(1))
                    candidates.append((epoch, fp))

        if not candidates:          # nothing matched
            return None

        # max by epoch index
        _, newest = max(candidates, key=lambda pair: pair[0])
        return newest

    parser = argparse.ArgumentParser(description="Test GateNet for aircraft pose estimation")

    parser.add_argument('--img-dir', type=str, required=True, help='Directory with images')
    parser.add_argument('--label-dir', type=str, required=True, help='Directory with pose labels')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Path to resume training from a checkpoint')
    
    args = parser.parse_args()

    script_dir = os.path.realpath(os.path.dirname(__file__))
    ckpt_folder_dir = os.path.join(script_dir, args.checkpoint_path)
    ckpt_dir = latest_checkpoint(ckpt_folder_dir)
    print(ckpt_dir)
    # ckpt_dir = os.path.join(script_dir, args.checkpoint_path, './checkpoint_epoch_30.pth')

    image_dir = os.path.join(script_dir, args.img_dir)
    poses_dir = os.path.join(script_dir, args.label_dir)
    img_fn = os.path.join(image_dir, f'img_{0}.png')
    img = Image.open(img_fn)
    img = np.array(img)
    img_shape = img.transpose(2,0,1).shape

    config = {
        'input_shape': img_shape,
        'output_shape': (3,),  # X, Y, Z, yaw, pitch, roll
        'l2_weight_decay': 1e-4,
        'batch_norm_decay': 0.99,
        'batch_norm_epsilon': 1e-3
    }
    model = GateNet(config = config)

    model.load_state_dict(torch.load(ckpt_dir)['model_state_dict'])

    model = model.to('cuda')

    images = []
    poses = []
    for i in range(200):
        img_fn = os.path.join(image_dir, f'img_{i}.png')
        pose_fn = os.path.join(poses_dir, f'img_{i}.txt')

        with open(pose_fn, 'r') as f:
            pose = f.read()

        pose = pose.strip('\n').strip(' ')
        pose = pose.split(' ')
        pose = [float(p) for p in pose]
        poses.append(pose)

        img = Image.open(img_fn)
        img = np.array(img)
        images.append(img)

    images = np.array(images)
    poses = np.array(poses)

    images_tensor = torch.FloatTensor(images).to('cuda')
    images_tensor = torch.permute(images_tensor, (0,3,1,2))
    poses_tensor = torch.FloatTensor(poses).to('cuda')

    estimated_poses = torch.zeros((0, 3)).to('cuda')
    with torch.no_grad():
        for i in range(0, 1000, 100):
            tmp = model(images_tensor[i:i+100])
            estimated_poses = torch.cat((estimated_poses, tmp), dim=0)
    # print(estimated_poses.shape)
    # print(torch.linalg.norm(estimated_poses-poses_tensor, dim=1))
    estimated_poses_array = estimated_poses.detach().cpu().numpy()
    poses_array = poses_tensor.detach().cpu().numpy()

    import matplotlib.pyplot as plt 
    plt.figure(0)
    plt.plot(poses_array[:,0], estimated_poses_array[:,0], 'b*')
    
    plt.figure(1)
    plt.plot(poses_array[:,1], estimated_poses_array[:,1], 'b*')
    
    plt.figure(2)
    plt.plot(poses_array[:,2], estimated_poses_array[:,2], 'b*')
    
    plt.show()