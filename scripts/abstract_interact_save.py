import os
import torch
import numpy as np
from PIL import Image
import argparse
import json
import base64
from io import BytesIO

def load_abstract_records(folder_path):
    records = []
    ref_images = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.startswith("abstract_") and file_name.endswith(".pt"):
            record = torch.load(os.path.join(folder_path, file_name))
            records.append(record)
        elif file_name.startswith("ref_") and file_name.endswith(".png"):
            ref_images.append(np.array(Image.open(os.path.join(folder_path, file_name))))
    return records, ref_images

def calculate_pose_bounds(records):
    poses = [rec["xl"] for rec in records] + [rec["xu"] for rec in records]
    pose_min = torch.min(torch.stack(poses), dim=0).values
    pose_max = torch.max(torch.stack(poses), dim=0).values
    perturbation_max = (pose_max - pose_min) / 4
    return pose_min, pose_max, perturbation_max

def filter_and_unify_images(records, camera_pose, perturbation_range):
    lower_images = []
    upper_images = []
    for rec in records:
        if torch.all(rec["xl"] >= camera_pose-perturbation_range) and torch.all(rec["xu"] <= camera_pose+perturbation_range):
            lower_images.append(rec["lower"].numpy())
            upper_images.append(rec["upper"].numpy())

    if not lower_images or not upper_images:  # Fallback if no images match the condition
        closest_record = min(records, key=lambda rec: torch.norm((rec["xl"] + rec["xu"]) / 2 - camera_pose))
        lower_images = [closest_record["lower"].numpy()]
        upper_images = [closest_record["upper"].numpy()]

    unified_lower = np.minimum.reduce(lower_images)
    unified_upper = np.maximum.reduce(upper_images)
    return unified_lower, unified_upper

def calculate_average_image(records, camera_pose, perturbation_range):
    """Calculate the average image for all images within the range of camera_pose ± perturbation_range."""
    ref_images = []
    for rec in records:
        if torch.all(rec["xl"] >= camera_pose - perturbation_range) and torch.all(rec["xu"] <= camera_pose + perturbation_range):
            ref_images.append(((rec["lower"]+rec["upper"])/2).numpy())  # Use "lower" images for averaging

    if not ref_images:  # Fallback if no images match the condition
        closest_record = min(records, key=lambda rec: torch.norm((rec["xl"] + rec["xu"]) / 2 - camera_pose))
        ref_images = [((closest_record["lower"]+closest_record["upper"])/2).numpy()]

    average_image = np.mean(ref_images, axis=0)
    return average_image

def find_closest_index(original_camera_poses, camera_pose):
    distances = [torch.norm(pose - camera_pose).item() for pose in original_camera_poses]
    return int(np.argmin(distances))

def encode_image_to_base64(image_array):
    """Convert a NumPy image array to a base64-encoded string."""
    image = Image.fromarray((image_array * 255).astype(np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save interactive visualization for abstract images as JSON.")
    parser.add_argument("--object_name", type=str, default="airplane_grey", help="Name of the object.")
    parser.add_argument("--domain_type", type=str, default="z", help="Domain type.")
    args = parser.parse_args()

    object_name = args.object_name
    domain_type = args.domain_type
    output_json = f"../Outputs/Analysis/{object_name}/{domain_type}/interactive_plot.json"
    folder_path = f"../Outputs/AbstractImages/{object_name}/{domain_type}"

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        exit(1)

    # Ensure the directory for the output JSON exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    records, ref_images = load_abstract_records(folder_path)
    pose_min, pose_max, perturbation_max = calculate_pose_bounds(records)

    # Convert pose_min and pose_max to 0-dimensional tensors
    pose_min = pose_min.item()
    pose_max = pose_max.item()

    # Restrict camera poses to 10 evenly spaced values
    restricted_pose_min = pose_min * 0.75 + pose_max * 0.25
    restricted_pose_max = pose_min * 0.25 + pose_max * 0.75
    camera_poses = torch.linspace(restricted_pose_min, restricted_pose_max, steps=10)

    # Restrict perturbation ranges to 10 evenly spaced values
    perturbation_ranges = torch.linspace(0, perturbation_max.max().item(), steps=10)

    # Precompute images for all slider values
    precomputed_images = {"lower": [], "upper": [], "ref": []}
    for camera_pose in camera_poses:
        for perturbation_range in perturbation_ranges:
            unified_lower, unified_upper = filter_and_unify_images(records, camera_pose, perturbation_range)
            average_ref_image = calculate_average_image(records, camera_pose, perturbation_range)

            precomputed_images["lower"].append(encode_image_to_base64(unified_lower))
            precomputed_images["upper"].append(encode_image_to_base64(unified_upper))
            precomputed_images["ref"].append(encode_image_to_base64(average_ref_image))

    # Save the interactive plot as a JSON file
    fig_dict = {
        "images": precomputed_images,
        "num_poses": len(camera_poses),
        "num_ranges": len(perturbation_ranges),
        "pose_values": camera_poses.tolist(),
        "range_values": perturbation_ranges.tolist()
    }

    with open(output_json, "w") as f:
        json.dump(fig_dict, f)

    print(f"Interactive plot saved to {output_json}")
