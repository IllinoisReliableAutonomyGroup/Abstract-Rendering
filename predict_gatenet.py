import os
import sys
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(grandfather_path)

from DownStreamModel.gatenet.gatenet import GateNet
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_model(config, weight_path):
    gatenet_config = {
        "input_shape": (3, config["height"], config["width"]),
        "output_shape": (3,),  # GateNet predicts 3D quantity
        "l2_weight_decay": 1e-4,
        "batch_norm_decay": 0.99,
        "batch_norm_epsilon": 1e-3,
    }
    model = GateNet(gatenet_config).to(DEVICE)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model

def preprocess_image(image_path, height, width):
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)

def predict_poses(model, img_dir, config):
    images = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith(".png")])
    predictions = []

    for img_path in images:
        image_tensor = preprocess_image(img_path, config["height"], config["width"])
        with torch.no_grad():
            predicted_pose = model(image_tensor).squeeze(0).cpu().numpy()
        predictions.append((os.path.basename(img_path), predicted_pose))

    return predictions

def main(config_path, img_dir):
    # Load configuration
    config = load_config(config_path)

    # Load model weights
    weights_dir = f"weights/{config['nn_type']}/{config['case_name']}"
    weight_file = os.path.join(weights_dir, "latest.pth")
    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"Weight file not found at {weight_file}")

    # Load model
    model = load_model(config, weight_file)

    # Predict poses
    script_dir = os.path.dirname(os.path.realpath(__file__))
    img_dir = os.path.join(script_dir, '../', img_dir)
    predictions = predict_poses(model, img_dir, config)

    # Print predictions
    print("Predicted Relative Poses:")
    for img_name, pose in predictions:
        print(f"{img_name}: {pose}")

if __name__ == "__main__":
    ### Default command: python3 scripts/predict_gatenet.py --config configs/${case_name}/gatenet.yml --img_dir Outputs/RenderedImages/${case_name}/cylinder/samples
    parser = argparse.ArgumentParser(description="Predict relative poses using GateNet.")
    parser.add_argument("--config", type=str, required=True, help="Path to the gatenet.yml configuration file.")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the directory containing input images.")
    args = parser.parse_args()

    main(args.config, args.img_dir)
