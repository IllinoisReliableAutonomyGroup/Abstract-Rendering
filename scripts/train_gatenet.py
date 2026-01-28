import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(grandfather_path)

from DownStreamModel.gatenet.gatenet import GateNet
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

class CustomDataset(Dataset):
    def __init__(self, image_dir, samples, transform=None):
        self.image_dir = image_dir
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, f"ref_{sample['index']:06d}.png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(sample["relative_pose"], dtype=torch.float32)
        return image, label

def main(config_file, samples_file):
    # ---------- Step 1: Load configuration ----------
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    with open(samples_file, "r") as f:
        samples = json.load(f)

    case_name = config["case_name"]
    odd_type = config["odd_type"]
    height, width = config["height"], config["width"]
    nn_type = config["nn_type"]
    img_folder = config.get("img_folder", None)

    image_dir = f"Outputs/RenderedImages/{case_name}/{odd_type}/{img_folder}" if img_folder else f"Outputs/Rendered Images/{case_name}/{odd_type}"

    # ---------- Step 2: Data loading ----------
    transform = transforms.Compose([
        transforms.Resize((height, width)),  # Resizes the image to (height, width)
        transforms.ToTensor(),  # Converts the image to a tensor with shape (3, height, width)
    ])
    dataset = CustomDataset(image_dir, samples, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ---------- Step 3: Model initialization ----------
    gatenet_config = {
        "input_shape": (3, height, width),
        "output_shape": (3,),  # GateNet predicts 3D quantity in the training script
        "l2_weight_decay": 1e-4,
        "batch_norm_decay": 0.99,
        "batch_norm_epsilon": 1e-3,
    }
    model = GateNet(gatenet_config).to(DEVICE)

    weights_dir = f"weights/{nn_type}/{case_name}"
    os.makedirs(weights_dir, exist_ok=True)
    weight_file = os.path.join(weights_dir, "latest.pth")
    if os.path.exists(weight_file):
        model.load_state_dict(torch.load(weight_file))
        print(f"Loaded weights from {weight_file}")

    # ---------- Step 4: Training setup ----------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 300

    # ---------- Step 5: Training ----------
    print("Training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        # Save weights every 50 epochs
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(weights_dir, f"epoch_{epoch}.pth"))
            print(f"Saved weights at epoch {epoch}")

    # Save final weights
    torch.save(model.state_dict(), weight_file)
    print(f"Final weights saved to {weight_file}")

    # ---------- Step 6: Post-training evaluation ----------
    print("Evaluating...")
    model.eval()
    total_error = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            error = torch.linalg.norm(outputs - labels, dim=1).mean().item()
            total_error += error

    avg_error = total_error / len(dataloader)
    print(f"Average error on training dataset: {avg_error:.4f}")

if __name__ == "__main__":
    ### export case_name="uturn"
    ### Default Command: python3 scripts/train_gatenet.py --config configs/${case_name}/gatenet.yml --samples configs/${case_name}/samples.json
    parser = argparse.ArgumentParser(description="Train GateNet model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the gatenet.yml configuration file.")
    parser.add_argument("--samples", type=str, required=True, help="Path to the samples.json file.")
    args = parser.parse_args()

    main(args.config, args.samples)
