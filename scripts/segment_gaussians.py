import os
import json
import sys
import argparse
import yaml

import numpy as np
import torch
from tqdm import tqdm
from collections import deque
from operator import itemgetter

from scipy.spatial.transform import Rotation
from PIL import Image

grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(grandfather_path)

from render_models import GsplatRGBOrigin, TransferModel

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32


def render_full_image(verf_net, input_ref, height, width, tile_size):
    """Render a full RGB image from the current pose."""
    img = np.zeros((height, width, 3))
    tiles_queue = deque(
        (h, w, min(h + tile_size, height), min(w + tile_size, width))
        for h in range(0, height, tile_size)
        for w in range(0, width, tile_size)
    )
    while tiles_queue:
        hl, wl, hu, wu = tiles_queue.popleft()
        tile_dict = {"hl": hl, "wl": wl, "hu": hu, "wu": wu}
        verf_net.call_model("update_tile", tile_dict)
        ref_tile = verf_net.forward(input_ref)
        ref_tile_np = ref_tile.squeeze(0).detach().cpu().numpy()
        img[hl:hu, wl:wu, :] = ref_tile_np
    img = (img.clip(0.0, 1.0) * 255).astype(np.uint8)
    return img


def render_segmented_image(verf_net, render_net, input_ref, labels, cluster_colors,
                           height, width, tile_size, min_distance, max_distance, gauss_num):
    """Render a flat segmentation image using the contribution map.
    Each pixel gets the solid color of its dominant Gaussian's object label."""
    pose_matrix = verf_net.preprocess(input_ref)
    sorted_depths = get_sorted_depths(render_net, pose_matrix)
    contribution_map, _ = get_gaussian_pixel_contributions(
        render_net, pose_matrix, height, width, tile_size, sorted_depths
    )

    # Map sorted indices back to original
    original_indices = get_sorted_to_original_map(
        render_net, pose_matrix, gauss_num, min_distance, max_distance
    )

    # Build flat color image (vectorized)
    cluster_colors_np = (cluster_colors.cpu().numpy() * 255).astype(np.uint8)
    img = np.full((height, width, 3), 128, dtype=np.uint8)  # gray for unlabeled

    valid = (contribution_map >= 0) & (contribution_map < len(original_indices))
    sorted_indices_valid = contribution_map[valid]
    orig_indices_valid = original_indices[sorted_indices_valid]
    pixel_labels = labels[orig_indices_valid].numpy()

    labeled = pixel_labels >= 0
    coords = np.argwhere(valid)  # [M, 2] array of (h, w)
    labeled_coords = coords[labeled]
    img[labeled_coords[:, 0], labeled_coords[:, 1]] = cluster_colors_np[pixel_labels[labeled]]

    return img


def get_gaussian_pixel_contributions(render_net, pose_matrix, height, width, tile_size, sorted_depths):
    """For each pixel, find which sorted Gaussian contributes most to the final color.
    Uses transmittance-weighted alpha (alpha * transmittance) so that front-facing
    Gaussians like gates get proper credit even if their raw alpha is lower.

    Returns:
        contribution_map: [H, W] sorted Gaussian index with highest weight per pixel
        depth_map: [H, W] depth of the dominant Gaussian per pixel
    """
    contribution_map = -1 * np.ones((height, width), dtype=np.int64)
    depth_map = np.zeros((height, width), dtype=np.float32)

    tiles_queue = deque(
        (h, w, min(h + tile_size, height), min(w + tile_size, width))
        for h in range(0, height, tile_size)
        for w in range(0, width, tile_size)
    )

    while tiles_queue:
        hl, wl, hu, wu = tiles_queue.popleft()
        tile_dict = {"hl": hl, "wl": wl, "hu": hu, "wu": wu}
        render_net.update_tile(tile_dict)

        alpha = render_net.render_alpha(pose_matrix, render_net.scene_dict_sorted)  # [1, TH, TW, N, 1]
        alpha = alpha.squeeze(0).squeeze(-1)  # [TH, TW, N]

        # Compute transmittance: T_i = prod(1 - alpha_j) for j < i
        alpha_shifted = torch.cat([torch.zeros_like(alpha[:, :, 0:1]), alpha[:, :, :-1]], dim=-1)
        transmittance = torch.cumprod(1.0 - alpha_shifted, dim=-1)  # [TH, TW, N]

        # Weight = alpha * transmittance (actual contribution to final pixel color)
        weight = alpha * transmittance  # [TH, TW, N]

        max_idx = weight.argmax(dim=-1)  # [TH, TW]
        contribution_map[hl:hu, wl:wu] = max_idx.cpu().numpy()

        # Record depth of dominant Gaussian
        max_idx_flat = max_idx.reshape(-1)
        depth_tile = sorted_depths[max_idx_flat].reshape(hu - hl, wu - wl)
        depth_map[hl:hu, wl:wu] = depth_tile.cpu().numpy()

    return contribution_map, depth_map


def get_sorted_depths(render_net, pose_matrix):
    """Get the depth of each sorted Gaussian from this viewpoint."""
    means_hom_world = render_net.scene_dict_sorted["means_hom_world"]
    means_hom_cam = torch.matmul(
        pose_matrix, means_hom_world[None, :, :].transpose(-1, -2)
    ).transpose(-1, -2)
    return means_hom_cam[0, :, 2]  # [N_sorted]


def split_masks_by_depth(masks, depth_map, contribution_map, sorted_depths, num_gaussians_sorted,
                         depth_threshold_ratio=0.3):
    """Split SAM masks into depth layers when Gaussians span a large depth range.
    This separates foreground objects (gates) from background (walls) within the same mask.

    Args:
        depth_threshold_ratio: fraction of total depth range to use as split threshold
    """
    split_masks = []

    for mask_dict in masks:
        seg = mask_dict['segmentation']  # [H, W] bool
        pixel_depths = depth_map[seg]

        if len(pixel_depths) == 0:
            split_masks.append(mask_dict)
            continue

        depth_range = pixel_depths.max() - pixel_depths.min()
        depth_median = np.median(pixel_depths)

        # If the depth range within this mask is large, split into foreground/background
        threshold = depth_threshold_ratio * (depth_map.max() - depth_map.min())

        if depth_range > threshold:
            # Split into foreground (closer) and background (farther)
            fg_mask = seg & (depth_map < depth_median)
            bg_mask = seg & (depth_map >= depth_median)

            if fg_mask.sum() > 50:  # min pixel count
                split_masks.append({
                    'segmentation': fg_mask,
                    'area': int(fg_mask.sum()),
                })
            if bg_mask.sum() > 50:
                split_masks.append({
                    'segmentation': bg_mask,
                    'area': int(bg_mask.sum()),
                })
        else:
            split_masks.append(mask_dict)

    return split_masks


def assign_labels_to_gaussians(contribution_map, masks, num_gaussians_sorted):
    """For each sorted Gaussian, vote for the SAM mask covering its dominant pixels."""
    num_masks = len(masks)
    votes = np.zeros((num_gaussians_sorted, num_masks), dtype=np.int32)

    for mask_idx, mask_dict in enumerate(masks):
        mask = mask_dict['segmentation']  # [H, W] bool
        gauss_indices = contribution_map[mask]
        valid = gauss_indices >= 0
        gauss_indices = gauss_indices[valid]

        if len(gauss_indices) > 0:
            unique, counts = np.unique(gauss_indices, return_counts=True)
            votes[unique, mask_idx] += counts

    return votes


def get_sorted_to_original_map(render_net, pose_matrix, gauss_num, min_distance, max_distance):
    """Get mapping from sorted Gaussian indices to original scene indices."""
    means_hom_world = render_net.scene_dict_all["means_hom_world"]
    means_hom_cam = torch.matmul(
        pose_matrix, means_hom_world[None, :, :].transpose(-1, -2)
    ).transpose(-1, -2)
    means_cam = means_hom_cam[:, :, :3]
    depth = means_cam[0, :, 2]
    mask_depth = (depth >= min_distance) & (depth <= max_distance)
    sorted_indices = torch.argsort(depth[mask_depth])

    all_original_indices = torch.arange(gauss_num, device=DEVICE)
    filtered_indices = all_original_indices[mask_depth]
    original_indices = filtered_indices[sorted_indices].cpu().numpy()
    return original_indices


def merge_small_objects(labels, means, min_object_size=50, merge_distance=0.15):
    """Merge small objects into their nearest large neighbor in 3D space.

    Args:
        labels: [N] tensor of object labels (-1 = unlabeled)
        means: [N, 3] tensor of Gaussian positions
        min_object_size: objects with fewer Gaussians than this get merged
        merge_distance: max 3D distance (centroid-to-centroid) to merge

    Returns:
        merged_labels: [N] tensor with small objects merged into neighbors
    """
    merged_labels = labels.clone()
    unique_labels = torch.unique(merged_labels[merged_labels >= 0])

    # Compute centroid and size for each object
    obj_info = {}
    for obj_id in unique_labels:
        mask = merged_labels == obj_id
        count = mask.sum().item()
        centroid = means[mask].mean(dim=0)
        obj_info[obj_id.item()] = {'count': count, 'centroid': centroid}

    # Separate into large and small objects
    large_objs = {k: v for k, v in obj_info.items() if v['count'] >= min_object_size}
    small_objs = {k: v for k, v in obj_info.items() if v['count'] < min_object_size}

    if len(large_objs) == 0:
        return merged_labels

    # Build tensor of large object centroids for distance computation
    large_ids = list(large_objs.keys())
    large_centroids = torch.stack([large_objs[k]['centroid'] for k in large_ids])  # [M, 3]

    # Merge each small object into nearest large neighbor
    for small_id, small_info in small_objs.items():
        dists = torch.norm(large_centroids - small_info['centroid'].unsqueeze(0), dim=1)
        min_dist, min_idx = dists.min(0)

        if min_dist.item() < merge_distance:
            target_id = large_ids[min_idx.item()]
            merged_labels[merged_labels == small_id] = target_id
            # Update the large object's centroid and count
            large_objs[target_id]['count'] += small_info['count']
        # else: leave it as its own small object

    return merged_labels


def make_side_by_side(rgb_img, seg_img):
    """Create a side-by-side image: [RGB | Segmentation]."""
    return np.concatenate([rgb_img, seg_img], axis=1)


def main(setup_dict):
    key_list = [
        "render_method", "case_name", "odd_type", "width", "height",
        "fx", "fy", "eps2d", "tile_size", "min_distance", "max_distance",
        "scene_path", "checkpoint_filename", "save_folder",
        "bg_img_path", "bg_pure_color", "poses", "sam_checkpoint"
    ]

    render_method, case_name, odd_type, width, height, fx, fy, eps2d, \
        tile_size, min_distance, max_distance, \
        scene_path, checkpoint_filename, save_folder, bg_img_path, \
        bg_pure_color, poses, sam_checkpoint = itemgetter(*key_list)(setup_dict)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    scene_folder = os.path.join(script_dir, scene_path)
    transform_file = os.path.join(scene_folder, 'dataparser_transforms.json')
    checkpoint_file = os.path.join(scene_folder, 'nerfstudio_models/', checkpoint_filename)

    # Load Transformation Matrix and Scale
    with open(transform_file, 'r') as fp:
        data_transform = json.load(fp)
        transform = np.array(data_transform['transform'])
        transform_hom = np.vstack((transform, np.array([0, 0, 0, 1])))
        scale_val = data_transform['scale']

    transform_hom = torch.from_numpy(transform_hom).to(dtype=DTYPE, device=DEVICE)
    scale = torch.tensor(scale_val, dtype=DTYPE, device=DEVICE)

    # Make output folder
    save_folder_full = os.path.join(script_dir, save_folder)
    os.makedirs(save_folder_full, exist_ok=True)

    # Load Trained 3DGS
    scene_parameters = torch.load(checkpoint_file, weights_only=False)
    means = scene_parameters['pipeline']['_model.gauss_params.means'].to(DEVICE)
    quats = scene_parameters['pipeline']['_model.gauss_params.quats'].to(DEVICE)
    opacities = torch.sigmoid(scene_parameters['pipeline']['_model.gauss_params.opacities']).to(DEVICE)
    scales = torch.exp(scene_parameters['pipeline']['_model.gauss_params.scales']).to(DEVICE)
    colors = torch.sigmoid(scene_parameters['pipeline']['_model.gauss_params.features_dc']).to(DEVICE)
    gauss_num = means.size(0)
    print(f"Total Gaussians in scene: {gauss_num}")

    camera_dict = {"fx": fx, "fy": fy, "width": width, "height": height}
    scene_dict_all = {
        "means": means, "quats": quats,
        "opacities": opacities, "scales": scales, "colors": colors
    }

    # Background
    if bg_img_path is None:
        bg_pure_color_t = torch.tensor(bg_pure_color)
        bg_color = bg_pure_color_t.view(1, 1, 3).repeat(height, width, 1).to(DEVICE)
    else:
        bg_img = Image.open(bg_img_path).convert("RGB")
        bg_img = bg_img.resize((height, width), Image.LANCZOS)
        bg_color = torch.from_numpy(np.array(bg_img, dtype=np.float32) / 255).to(DEVICE)

    # Prepare Poses
    transs, orientations = np.array(poses)[:, :3], np.array(poses)[:, 3:]
    Rs = Rotation.from_euler('ZYX', orientations)
    rots = Rs.as_matrix()

    # Setup renderer
    pipeline_type = "rendering"
    render_net = GsplatRGBOrigin(camera_dict, scene_dict_all, min_distance, max_distance, bg_color, eps2d).to(DEVICE)
    verf_net = TransferModel(render_net, pipeline_type, None, None, transform_hom, scale, odd_type).to(DEVICE)

    # Load SAM
    print("Loading SAM model...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,
    )
    print("SAM model loaded.")

    # ---- Phase 1: Collect per-Gaussian labels from each view ----
    # Each Gaussian gets a globally unique label from its best mask in each view
    gaussian_global_labels = {}  # orig_gauss_idx -> list of global_label
    rendered_images = []
    global_label_offset = 0

    num_views = len(transs)
    print(f"\n=== Phase 1: Segmenting {num_views} views ===")

    for view_idx in tqdm(range(num_views), desc="Segmenting views"):
        base_trans = torch.from_numpy(transs[view_idx]).to(device=DEVICE, dtype=DTYPE)
        rot = torch.from_numpy(rots[view_idx]).to(device=DEVICE, dtype=DTYPE)
        verf_net.update_model_param(rot, base_trans)

        input_ref = torch.zeros((1, 3)).to(device=DEVICE, dtype=DTYPE)

        # Sort Gaussians for this view
        verf_net.call_model_preprocess("sort_gauss", input_ref)

        # Render RGB image
        img = render_full_image(verf_net, input_ref, height, width, tile_size)
        rendered_images.append(img)

        # Run SAM
        masks = mask_generator.generate(img)
        if len(masks) == 0:
            print(f"  View {view_idx}: No masks found, skipping.")
            continue

        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        print(f"  View {view_idx}: Found {len(masks)} masks")

        # Get Gaussian-to-pixel contribution map with depth
        pose_matrix = verf_net.preprocess(input_ref)
        sorted_depths = get_sorted_depths(render_net, pose_matrix)
        contribution_map, depth_map = get_gaussian_pixel_contributions(
            render_net, pose_matrix, height, width, tile_size, sorted_depths
        )

        # Split SAM masks by depth to separate foreground (gates) from background
        masks = split_masks_by_depth(
            masks, depth_map, contribution_map, sorted_depths,
            len(sorted_depths), depth_threshold_ratio=0.15
        )
        print(f"  View {view_idx}: {len(masks)} masks after depth splitting")

        # Map sorted indices back to original
        original_indices = get_sorted_to_original_map(
            render_net, pose_matrix, gauss_num, min_distance, max_distance
        )
        num_sorted = len(original_indices)

        # Get votes in sorted space, then map to original
        votes_sorted = assign_labels_to_gaussians(contribution_map, masks, num_sorted)

        for sorted_idx in range(num_sorted):
            orig_idx = original_indices[sorted_idx]
            best_mask = votes_sorted[sorted_idx].argmax()
            vote_count = votes_sorted[sorted_idx, best_mask]
            if vote_count > 0:
                # Use globally unique label: offset + per-view mask index
                global_label = global_label_offset + int(best_mask)
                if orig_idx not in gaussian_global_labels:
                    gaussian_global_labels[orig_idx] = []
                gaussian_global_labels[orig_idx].append(global_label)

        global_label_offset += len(masks)

    # ---- Phase 2: Consolidate labels via majority vote ----
    # Each Gaussian was assigned a global label from each view it appeared in.
    # We pick the label that was assigned most frequently (majority vote).
    # Then we cluster: Gaussians sharing the same global label are the same object.
    print("\n=== Phase 2: Consolidating labels ===")

    from collections import Counter
    labels = -1 * torch.ones(gauss_num, dtype=torch.int64)
    for orig_idx, glabels in gaussian_global_labels.items():
        # Pick the most common global label across views
        most_common = Counter(glabels).most_common(1)[0][0]
        labels[orig_idx] = most_common

    num_labeled = (labels >= 0).sum().item()
    unique_labels_pre = torch.unique(labels[labels >= 0])
    num_objects_pre = len(unique_labels_pre)
    print(f"Labeled {num_labeled}/{gauss_num} Gaussians into {num_objects_pre} objects (before merge)")

    # Merge small objects into nearest large neighbor in 3D
    # Iteratively merge small objects until stable
    for merge_round in range(3):
        labels = merge_small_objects(labels, means, min_object_size=2000, merge_distance=1.0)
    unique_labels = torch.unique(labels[labels >= 0])
    num_objects = len(unique_labels)
    print(f"After spatial merge: {num_objects} objects (merged {num_objects_pre - num_objects} small objects)")

    # Compute centroids per object (mean position of Gaussians in each cluster)
    centroids = torch.zeros((num_objects, 3), dtype=torch.float32)
    for i, obj_id in enumerate(unique_labels):
        obj_mask = labels == obj_id
        centroids[i] = means[obj_mask].mean(dim=0).cpu()

    # Generate perceptually distinct cluster colors using evenly spaced hues
    import colorsys
    cluster_colors_np = np.zeros((num_objects, 3), dtype=np.float32)
    golden_ratio = 0.618033988749895
    h = 0.0
    for i in range(num_objects):
        # Use golden ratio to space hues, vary saturation/value for more distinction
        s = 0.7 + 0.3 * ((i % 3) / 2.0)     # saturation: 0.7, 0.85, 1.0
        v = 0.95 - 0.2 * ((i % 5) / 4.0)     # value: 0.95, 0.90, 0.85, 0.80, 0.75
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cluster_colors_np[i] = [r, g, b]
        h = (h + golden_ratio) % 1.0
    cluster_colors = torch.from_numpy(cluster_colors_np)

    # Remap labels to be contiguous 0..num_objects-1
    label_remap = {int(old_label): new_idx for new_idx, old_label in enumerate(unique_labels.tolist())}
    labels_contiguous = labels.clone()
    for old_label, new_idx in label_remap.items():
        labels_contiguous[labels == old_label] = new_idx

    # Save cluster_assignments.pt (matching existing format)
    cluster_result = {
        "labels": labels_contiguous,
        "centroids": centroids,
        "cluster_colors": cluster_colors,
    }
    cluster_path = os.path.join(save_folder_full, "cluster_assignments.pt")
    torch.save(cluster_result, cluster_path)
    print(f"Saved cluster assignments to {cluster_path}")

    # ---- Phase 3: Render side-by-side images ----
    print(f"\n=== Phase 3: Rendering {num_views} side-by-side images ===")

    cluster_colors_device = cluster_colors.to(DEVICE)

    for view_idx in tqdm(range(num_views), desc="Rendering segmented views"):
        base_trans = torch.from_numpy(transs[view_idx]).to(device=DEVICE, dtype=DTYPE)
        rot = torch.from_numpy(rots[view_idx]).to(device=DEVICE, dtype=DTYPE)
        verf_net.update_model_param(rot, base_trans)

        input_ref = torch.zeros((1, 3)).to(device=DEVICE, dtype=DTYPE)
        verf_net.call_model_preprocess("sort_gauss", input_ref)

        # Get the RGB image (already rendered)
        rgb_img = rendered_images[view_idx]

        # Render segmentation-colored image
        seg_img = render_segmented_image(
            verf_net, render_net, input_ref, labels_contiguous, cluster_colors_device,
            height, width, tile_size, min_distance, max_distance, gauss_num
        )

        # Save side-by-side
        combined = make_side_by_side(rgb_img, seg_img)
        Image.fromarray(combined).save(os.path.join(save_folder_full, f'seg_{view_idx:06d}.png'))

    # Print summary
    for i, obj_id in enumerate(unique_labels):
        obj_count = (labels == obj_id).sum().item()
        print(f"  Object {i}: {obj_count} Gaussians")

    print(f"\nDone! Output saved to {save_folder_full}")
    return labels_contiguous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segment Gaussians by objects using SAM.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--odd", type=str, required=True, help="Path to the JSON trajectory file.")
    parser.add_argument("--sam_checkpoint", type=str,
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                             "../weights/sam_vit_h_4b8939.pth"),
                        help="Path to SAM checkpoint.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.odd, 'r') as f:
        odd_file = json.load(f)

    scene_path = f"../nerfstudio/outputs/{config['case_name']}/{config['render_method']}/{config['data_time']}"
    save_folder = f"../Outputs/SegmentedImages/{config['case_name']}/{config['odd_type']}"

    setup_dict = {
        "render_method": config["render_method"],
        "case_name": config["case_name"],
        "odd_type": config["odd_type"],
        "width": config["width"],
        "height": config["height"],
        "fx": config["fx"],
        "fy": config["fy"],
        "eps2d": config["eps2d"],
        "tile_size": config["tile_size_render"],
        "min_distance": config["min_distance"],
        "max_distance": config["max_distance"],
        "scene_path": scene_path,
        "checkpoint_filename": config["checkpoint_filename"],
        "save_folder": save_folder,
        "bg_img_path": config["bg_img_path"],
        "bg_pure_color": config["bg_pure_color"],
        "poses": [odd["pose"] for odd in odd_file],
        "sam_checkpoint": args.sam_checkpoint,
    }

    main(setup_dict)
