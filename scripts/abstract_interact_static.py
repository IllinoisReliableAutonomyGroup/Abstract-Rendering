import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse
import mpld3  # Import mpld3 for saving interactive plots
import json

def load_abstract_records(folder_path):
    records = []
    ref_images = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.startswith("abstract_") and file_name.endswith(".pt"):
            file_path = os.path.join(folder_path, file_name)
            record = torch.load(file_path)
            records.append(record)
        elif file_name.startswith("ref_") and file_name.endswith(".png"):
            ref_path = os.path.join(folder_path, file_name)
            ref_images.append(np.array(Image.open(ref_path)))
    return records, ref_images

def calculate_pose_bounds(records):
    pose_min = torch.min(torch.stack([rec["xl"] for rec in records]), dim=0).values
    pose_max = torch.max(torch.stack([rec["xu"] for rec in records]), dim=0).values
    perturbation_max = (pose_max - pose_min) / 4
    return pose_min, pose_max, perturbation_max

def filter_and_unify_images(records, camera_pose, perturbation_range):
    lower_images = []
    upper_images = []
    for rec in records:
        if torch.all(rec["xu"] > (camera_pose - perturbation_range)) and torch.all(rec["xl"] < (camera_pose + perturbation_range)):
            lower_images.append(rec["lower"].numpy())
            upper_images.append(rec["upper"].numpy())
    if not lower_images or not upper_images:  # Select closest image if no matches
        closest_record = min(records, key=lambda rec: torch.norm((rec["xl"] + rec["xu"]) / 2 - camera_pose))
        lower_images = [closest_record["lower"].numpy()]
        upper_images = [closest_record["upper"].numpy()]
    unified_lower = np.minimum.reduce(lower_images)
    unified_upper = np.maximum.reduce(upper_images)
    return unified_lower, unified_upper

def find_closest_index(original_camera_poses, camera_pose):
    closest_index = min(range(len(original_camera_poses)), key=lambda i: torch.norm(original_camera_poses[i] - camera_pose))
    return closest_index

def update_display(val):
    pose_idx = int(slider_pose.val * (len(camera_poses) - 1))
    range_idx = int(slider_range.val * max_range_idx)
    camera_pose = camera_poses[pose_idx]
    perturbation_range = range_idx * min_perturbation_range
    unified_lower, unified_upper = filter_and_unify_images(records, camera_pose, perturbation_range)
    closest_index = find_closest_index(original_camera_poses, camera_pose)  # Compare with original camera poses
    ax1.imshow((unified_lower * 255).astype(np.uint8))
    ax2.imshow((unified_upper * 255).astype(np.uint8))
    ax3.imshow(ref_images[closest_index])  # Display the correct ref image
    fig.canvas.draw_idle()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive visualization for abstract images.")
    parser.add_argument("--object_name", type=str, default="airplane_grey", help="Name of the object.")
    parser.add_argument("--domain_type", type=str, default="pitch", help="Domain type.")
    args = parser.parse_args()

    object_name = args.object_name
    domain_type = args.domain_type
    output_json = f"../Outputs/Analysis/{object_name}/{domain_type}/interactive_plot.json"
    folder_path = f"../Outputs/AbstractImages/{object_name}/{domain_type}"

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        exit(1)

    records, ref_images = load_abstract_records(folder_path)
    pose_min, pose_max, perturbation_max = calculate_pose_bounds(records)

    # Restrict camera poses to the range (pose_min * 0.75 + pose_max * 0.25) to (pose_min * 0.25 + pose_max * 0.75)
    restricted_pose_min = pose_min * 0.75 + pose_max * 0.25
    restricted_pose_max = pose_min * 0.25 + pose_max * 0.75
    original_camera_poses = [(rec["xl"] + rec["xu"]) / 2 for rec in records]  # Original camera poses
    camera_poses = [pose for pose in original_camera_poses if torch.all(pose >= restricted_pose_min) and torch.all(pose <= restricted_pose_max)]

    # Calculate minimum perturbation range
    min_perturbation_range = torch.mean(torch.stack(
        [torch.norm(records[i + 1]["xl"] - records[i]["xl"]) for i in range(len(records) - 1)]
    )).item()
    max_perturbation_range = (pose_max - pose_min).max().item() / 4
    max_range_idx = int(max_perturbation_range // min_perturbation_range)

    # Initial values for sliders
    initial_pose_idx = len(camera_poses) // 2
    initial_range_idx = 0  # Start with perturbation range of 0

    # Create interactive plot
    fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(15, 5))  # Add a middle axis for the ref image
    plt.subplots_adjust(bottom=0.25)

    unified_lower, unified_upper = filter_and_unify_images(records, camera_poses[initial_pose_idx], 0)
    closest_index = find_closest_index(original_camera_poses, camera_poses[initial_pose_idx])  # Use original camera poses
    img1 = ax1.imshow((unified_lower * 255).astype(np.uint8))
    img2 = ax2.imshow((unified_upper * 255).astype(np.uint8))
    img3 = ax3.imshow(ref_images[closest_index])  # Display the initial ref image
    print("Initial closest index:", closest_index)
    ax1.set_title("Lower Bound")
    ax2.set_title("Upper Bound")
    ax3.set_title("Ref Image")

    # Add sliders
    ax_slider_pose = plt.axes([0.2, 0.1, 0.65, 0.03])
    ax_slider_range = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider_pose = Slider(ax_slider_pose, "Camera Pose", 0, 1, valinit=initial_pose_idx / (len(camera_poses) - 1))
    slider_range = Slider(ax_slider_range, "Perturbation Range", 0, 1, valinit=initial_range_idx / max_range_idx)

    slider_pose.on_changed(update_display)
    slider_range.on_changed(update_display)

    # Ensure the directory for the output JSON exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    # Save the interactive plot as a JSON file
    fig_dict = mpld3.fig_to_dict(fig)
    with open(output_json, "w") as f:
        json.dump(fig_dict, f)

    print(f"Interactive plot saved to {output_json}")

    # Show the plot
    # plt.show()