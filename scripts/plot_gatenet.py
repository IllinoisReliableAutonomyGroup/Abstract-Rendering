import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse

from utils_rotation import orientation_to_direction
from utils_transform import orthogonal_basis_from_direction
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

def load_analysis_results(case_name, odd_type, nn_type):
    result_path = Path(f"Outputs/Analysis/{case_name}/{odd_type}/{nn_type}_result.pt")
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found at {result_path}")
    return torch.load(result_path)

def generate_gate_circle(point, tangent, outer_radius=0.2, height=0.1, num_points=100, num_slices=5):
    """
    Generate a hollow cylindrical region representing a gate with multiple circles between the top and bottom surfaces.

    Args:
        point (list): Center of the gate [x, y, z].
        tangent (list): Tangent vector [dx, dy, dz] defining the gate's orientation.
        outer_radius (float): Outer radius of the gate.
        height (float): Height of the cylindrical region.
        num_points (int): Number of points to generate the circle.
        num_slices (int): Number of slices between the top and bottom surfaces.

    Returns:
        np.ndarray: Points representing the hollow cylindrical region.
    """
    inner_radius = outer_radius * 0.85  # Define inner radius for hollow effect
    # Normalize tangent
    tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

    # Find two orthogonal vectors to the tangent
    if np.allclose(tangent, [1, 0, 0]) or np.allclose(tangent, [-1, 0, 0]):
        orthogonal1 = np.array([0, 1, 0])
    else:
        orthogonal1 = np.cross(tangent, [1, 0, 0])
        orthogonal1 = orthogonal1 / np.linalg.norm(orthogonal1)

    orthogonal2 = np.cross(tangent, orthogonal1)

    # Generate points for the outer and inner circles
    theta = np.linspace(0, 2 * np.pi, num_points)
    outer_circle = (
        outer_radius * np.outer(np.cos(theta), orthogonal1) +
        outer_radius * np.outer(np.sin(theta), orthogonal2)
    )
    inner_circle = (
        inner_radius * np.outer(np.cos(theta), orthogonal1) +
        inner_radius * np.outer(np.sin(theta), orthogonal2)
    )

    # Generate slices between the top and bottom surfaces
    slice_positions = np.linspace(-height / 2, height / 2, num_slices)
    hollow_cylinder = []

    for z in slice_positions:
        hollow_cylinder.append(outer_circle + np.array(point) + z * tangent)
        hollow_cylinder.append(inner_circle + np.array(point) + z * tangent)

    # Combine all slices into a single array
    hollow_cylinder = np.vstack(hollow_cylinder)

    return hollow_cylinder


def plot_filled_cylinder_fast(
    ax, base, direction, radius, xl, xu, color, label=None
):
    """
    Plot a filled cylinder in 3D space.

    Parameters:
    - ax: Matplotlib 3D axis.
    - base: Base point of the cylinder.
    - direction: Direction vector of the cylinder.
    - radius: Radius of the cylinder.
    - xl, xu: Bounds for the cylinder.
    - color: Color of the cylinder.
    - label: Label for the legend.
    """
    d0, r0, a0 = xl
    d1, r1, a1 = xu

    # Aggressively reduced resolution
    n_d = 2
    n_theta = 20
    n_r = 5

    d = np.linspace(d0, d1, n_d)
    theta = np.linspace(a0, a1, n_theta)
    r_vals = np.linspace(r0, r1, n_r)

    d, theta = np.meshgrid(d, theta, indexing="ij")

    direction_t = torch.as_tensor(direction, dtype=torch.float32)
    base_t = torch.as_tensor(base, dtype=torch.float32)
    radius_t = torch.as_tensor(radius, dtype=torch.float32)

    _, o1, o2 = orthogonal_basis_from_direction(direction_t)

    d_t = torch.from_numpy(d).float()
    theta_t = torch.from_numpy(theta).float()

    alpha = 0.35 / n_r  # keep opacity stable

    for r in r_vals:
        points = (
            base_t[None, None, :]
            + d_t[..., None] * direction_t
            + (radius_t * r)
              * (
                  torch.cos(theta_t)[..., None] * o1
                  + torch.sin(theta_t)[..., None] * o2
              )
        )

        X = points[..., 0].numpy()
        Y = points[..., 1].numpy()
        Z = points[..., 2].numpy()

        ax.plot_surface(
            X, Y, Z,
            color=color,
            alpha=alpha,
            linewidth=0,
            antialiased=False,
            label=label
        )

def plot_gate_poses(ax, gate_poses, gate_radius=0.5):
    """
    Plot gate poses as circles in 3D space.

    Parameters:
    - ax: Matplotlib 3D axis.
    - gate_poses: List of gate poses, each as [x, y, z, yaw, pitch, roll].
    - gate_radius: Radius of the gate circles.
    """
    for i, gate_pose in enumerate(gate_poses):
        gate_point = gate_pose[:3]
        yaw, pitch, roll = gate_pose[3:]
        tangent = orientation_to_direction(yaw, pitch, roll)
        circle = generate_gate_circle(gate_point, tangent, gate_radius)
        ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], color="blue", label="Gate" if i == 0 else None)

from matplotlib.patches import Patch

def main(config_path, traj_path):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    case_name = config["case_name"]
    odd_type = config["odd_type"]
    nn_type = config["nn_type"]
    threshold = np.array(config["threshold"])
    debug = config["debug"]
    show_details = config.get("show_details", False)  # Read the new variable

    # Load trajectory data
    with open(traj_path, "r") as f:
        traj_data = yaml.safe_load(f)
    gate_poses = traj_data["gate_poses"]

    # Load analysis results
    results = load_analysis_results(case_name, odd_type, nn_type)
    points = results["point"].numpy()
    directions = results["direction"].numpy()
    radii = results["radius"].numpy()
    xl = results["xl"].numpy()
    xu = results["xu"].numpy()
    out_lb = results["out_lb"].numpy()
    out_ub = results["out_ub"].numpy()
    out_center = results["out_center"].numpy()

    # Replace NaN values with infinity
    out_lb = np.nan_to_num(out_lb, nan=-np.inf)
    out_ub = np.nan_to_num(out_ub, nan=np.inf)

    # Define a custom colormap for cyan to magenta
    cyan_magenta_cmap = LinearSegmentedColormap.from_list("cyan_magenta_gradient", ["#00FFFF", "#FFFF00"])

    # Compute bounds_diff and its min/max for normalization
    bounds_diff = out_ub - out_lb
    if show_details:
        bounds_diff_norms = np.linalg.norm(bounds_diff, axis=1)
        bounds_diff_min = np.min(bounds_diff_norms)
        bounds_diff_max = np.max(bounds_diff_norms)

        # Normalize bounds_diff_norms to [bounds_diff_min / bounds_diff_max, 1]
        normalized_bounds_diff_min = bounds_diff_min / bounds_diff_max
        normalized_bounds_diff_max = 1.0

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Track if labels for legend have been added
    success_label_added = False
    failure_label_added = False

    # Plot each data point as a part of a cylinder
    for i in range(points.shape[0]):
        base = points[i]
        direction = directions[i]
        radius = radii[i]
        coeff_lb = xl[i]
        coeff_ub = xu[i]

        # Determine the color based on the threshold or bounds_diff norm
        if show_details:
            norm = np.linalg.norm(bounds_diff[i])
            normalized_value = (norm - bounds_diff_min) / (bounds_diff_max - bounds_diff_min + 1e-8)
            normalized_value = normalized_bounds_diff_min + normalized_value * (normalized_bounds_diff_max - normalized_bounds_diff_min)
            color = cyan_magenta_cmap(normalized_value)  # Map to custom cyan-to-magenta colormap
        else:
            if np.all(bounds_diff[i] < threshold):
                cololabel = "Success" if not success_label_added else None
                success_label_added = True
                color = "green"
                label = "Success" if not success_label_added else None
                success_label_added = True
            else:
                color = "red"
                label = "Fail" if not failure_label_added else None
                failure_label_added = True

        if debug:
            print(f"Point {i}: bounds_diff = {bounds_diff[i]}, color = {color}")

        # Plot the cylinder
        plot_filled_cylinder_fast(ax, base, direction, radius, coeff_lb, coeff_ub, color=color)

    # Plot gate poses
    plot_gate_poses(ax, gate_poses, gate_radius=0.5)

    # Compress the z-axis to 1/3 of its original range
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    ax.set_box_aspect([1, 1, (z_limits[1] - z_limits[0]) / 3])

    # Do not show color bar
    # Save the figure
    output_dir = Path(f"Outputs/Analysis/{case_name}/{odd_type}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "certification_visualization.png"
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {output_file}")

    plt.show()

if __name__ == "__main__":
    ### Define command: python3 scripts/plot_gatenet.py --config configs/${case_name}/gatenet.yml --traj configs/${case_name}/traj.yaml
    parser = argparse.ArgumentParser(description="Plot GateNet analysis results and gates.")
    parser.add_argument("--config", type=str, required=True, help="Path to the gatenet.yml configuration file.")
    parser.add_argument("--traj", type=str, required=True, help="Path to the trajectory YAML file.")
    # parser.add_argument("--threshold", type=float, nargs=3, default=[120, 150, 90], help="Threshold for coloring cylinders.")
    args = parser.parse_args()

    main(args.config, args.traj)
