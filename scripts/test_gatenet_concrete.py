import os
import sys
import json
import argparse

import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D projection

try:
	import plotly.graph_objects as go  # type: ignore
	HAS_PLOTLY = True
except ImportError:
	HAS_PLOTLY = False


# Make project root importable (DownStreamModel, etc.)
grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
if grandfather_path not in sys.path:
	sys.path.append(grandfather_path)

from DownStreamModel.gatenet.gatenet import GateNet  # noqa: E402
from plot_gatenet import plot_gate_poses


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConcreteImageDataset(Dataset):
	"""Dataset for concrete images + pose labels (for testing)."""

	def __init__(self, samples_json, image_root, target_size=(64, 64)):
		with open(samples_json, "r") as f:
			all_samples = json.load(f)
		self.image_root = image_root

		self.transform = transforms.Compose(
			[
				transforms.Resize(target_size),
				transforms.ToTensor(),
			]
		)

		skipped = sum(1 for s in all_samples if not s.get("include", True))
		self.samples = [s for s in all_samples if s.get("include", True)]
		print(f"Loaded {len(self.samples)} samples from {samples_json} ({skipped} skipped — include=False)")

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]

		img_filename = f"ref_{sample['index']:06d}.png"
		img_path = os.path.join(self.image_root, img_filename)

		img = Image.open(img_path).convert("RGB")
		img = self.transform(img)

		pose = torch.tensor(sample["relative_pose"], dtype=torch.float32)
		world_pos = torch.tensor(sample["pose"][:3], dtype=torch.float32)

		return img, pose, world_pos


def build_gatenet_from_config(config: dict) -> torch.nn.Module:
	"""Construct GateNet with the same shape params used during training."""

	model_config = {
		"input_shape": (3, config["image_height"], config["image_width"]),
		"output_shape": (3,),
		"batch_norm_decay": config["batch_norm_decay"],
		"batch_norm_epsilon": config["batch_norm_epsilon"],
	}
	model = GateNet(model_config)
	return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
	ckpt = torch.load(ckpt_path, map_location=device)
	state_dict = ckpt.get("model_state_dict", ckpt)
	# Strip off any 'module.' prefixes in case of DataParallel
	clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
	model.load_state_dict(clean_state_dict)
	return model


def evaluate_concrete(config_path: str, traj_path: str = None):
	# Load train+certify config (reused here for paths and shapes)
	with open(config_path, "r") as f:
		config = yaml.safe_load(f)

	print("=== Test config ===")
	for k, v in config.items():
		print(f"{k}: {v}")

	# Optional: timestamped run directory for checkpoints/results
	run_datetime = config.get("run_datetime")

	# Dataset
	dataset = ConcreteImageDataset(
		samples_json=config["concrete_samples_json"],
		image_root=config["concrete_image_root"],
		target_size=(config["image_width"], config["image_height"]),
	)

	dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

	# Model + checkpoint
	model = build_gatenet_from_config(config).to(DEVICE)

	base_ckpt_dir = config["checkpoint_dir"]
	if run_datetime is not None:
		ckpt_dir = os.path.join(base_ckpt_dir, run_datetime)
		ckpt_path = os.path.join(ckpt_dir, "latest.pth")
	else:
		ckpt_dir = base_ckpt_dir
		ckpt_path = os.path.join(ckpt_dir, "latest.pth")
		print(
			"[Warning] 'run_datetime' not set in config; "
			"loading checkpoint from legacy path: " f"{ckpt_path}"
		)

	if not os.path.exists(ckpt_path):
		raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

	model = load_checkpoint(model, ckpt_path, DEVICE)
	model.eval()

	print(f"Loaded GateNet checkpoint from: {ckpt_path}")

	# Collect predictions and ground truth
	true_poses = []
	pred_poses = []
	world_positions = []

	with torch.no_grad():
		for imgs, poses, world_pos in dataloader:
			imgs = imgs.to(DEVICE)
			poses = poses.to(DEVICE)

			outputs = model(imgs)

			true_poses.append(poses.squeeze(0).cpu().numpy())
			pred_poses.append(outputs.squeeze(0).cpu().numpy())
			world_positions.append(world_pos.squeeze(0).cpu().numpy())

	true_poses = np.stack(true_poses, axis=0)      # [N, 3]
	pred_poses = np.stack(pred_poses, axis=0)      # [N, 3]
	world_positions = np.stack(world_positions, axis=0)  # [N, 3]

	# Error vectors and norms
	errors = pred_poses - true_poses
	error_norms = np.linalg.norm(errors, axis=1)  # [N]
	# Clip norms for visualization in a fixed range [0, 1.25]
	error_norms_clipped = np.clip(error_norms, 0.0, 0.5)

	print("=== Summary ===")
	print(f"Num samples: {true_poses.shape[0]}")
	print(f"Mean error norm: {error_norms.mean():.6f}")
	print(f"Max  error norm: {error_norms.max():.6f}")

	# Save vectors for further analysis
	out_dir = ckpt_dir
	os.makedirs(out_dir, exist_ok=True)
	npz_path = os.path.join(out_dir, "concrete_test_poses.npz")
	np.savez(npz_path, true_poses=true_poses, pred_poses=pred_poses, error_norms=error_norms)
	print(f"Saved poses and error norms (npz) to: {npz_path}")

	# Also save a simple text file: one line per sample
	txt_path = os.path.join(out_dir, "concrete_test_poses.txt")
	with open(txt_path, "w") as f:
		f.write("# idx  true_x  true_y  true_z  pred_x  pred_y  pred_z  err_norm\n")
		for idx, (t, p, e) in enumerate(zip(true_poses, pred_poses, error_norms)):
			f.write(
				f"{idx:04d}  "
				f"{t[0]:.6f}  {t[1]:.6f}  {t[2]:.6f}  "
				f"{p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}  "
				f"{e:.6f}\n"
			)
	print(f"Saved poses and error norms (txt) to: {txt_path}")

	# 3D plot: samples at world positions, colored by error norm
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")

	xs = world_positions[:, 0]
	ys = world_positions[:, 1]
	zs = world_positions[:, 2]

	# Color of each point encodes error norm (clipped to [0, 1.25])
	sc = ax.scatter(
		xs,
		ys,
		zs,
		c=error_norms_clipped,
		cmap="plasma",
		vmin=0.0,
		vmax=0.5,
		s=10,
	)

	cbar = fig.colorbar(sc, ax=ax, pad=0.1)
	cbar.set_label("error norm (clipped to [0, 0.5])")

	# Plot gates from traj.yaml if provided
	if traj_path is not None:
		with open(traj_path, "r") as f:
			traj_data = yaml.safe_load(f)
		gate_poses = traj_data.get("gate_poses", [])
		gate_radius = traj_data.get("gate_radius", 0.5)
		plot_gate_poses(ax, gate_poses, gate_radius=gate_radius)

	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	ax.set_title("Concrete GateNet error norms (color = L2 error, 0–0.5)")

	plot_name = f"concrete_errors_{config['bound_method']}_lambda{config['lambda_concrete']}.png"
	plot_path = os.path.join(out_dir, plot_name)
	plt.savefig(plot_path, bbox_inches="tight")
	print(f"Saved 3D error plot to: {plot_path}")

	# Show interactive 3D window (rotate/zoom) when run locally
	plt.show()
	plt.close(fig)

	# Optional: interactive 3D plot (HTML) if plotly is available
	if HAS_PLOTLY:
		fig_int = go.Figure()
		# World positions as scatter with color = clipped error norm
		fig_int.add_trace(
			go.Scatter3d(
				x=xs,
				y=ys,
				z=zs,
				mode="markers",
				marker=dict(
					size=4,
					color=error_norms_clipped,
					colorscale="Plasma",
					cmin=0.0,
					cmax=1.25,
					colorbar=dict(title="error norm (clipped to [0, 0.5])"),
				),
				name="true pose",
			)
		)

		fig_int.update_layout(
			scene=dict(
				xaxis_title="x",
				yaxis_title="y",
				zaxis_title="z",
			),
			title="Concrete GateNet error norms (color = L2 error, 0–0.5)",
		)

		html_name = (
			f"concrete_errors_{config['bound_method']}_lambda{config['lambda_concrete']}_interactive.html"
		)
		html_path = os.path.join(out_dir, html_name)
		fig_int.write_html(html_path)
		print(f"Saved interactive 3D error plot to: {html_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Test GateNet on concrete images and visualize pose errors."
	)
	parser.add_argument(
		"--config",
		type=str,
		required=True,
		help="Path to training config YAML (e.g., configs/mini_line/train_certify_config.yml)",
	)
	parser.add_argument(
		"--traj",
		type=str,
		default=None,
		help="Path to traj.yaml for gate poses (e.g., configs/mini_line/traj.yaml)",
	)
	args = parser.parse_args()

	evaluate_concrete(args.config, args.traj)

