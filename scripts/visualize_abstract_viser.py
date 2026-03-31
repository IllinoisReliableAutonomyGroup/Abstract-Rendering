#!/usr/bin/env python3
"""
Visualize abstract certification cuboids in Viser alongside the Gaussian scene.

Usage:
    python3 scripts/visualize_abstract_viser.py \
        --config configs/<case_name>/train_certify_config.yml [--option normal|ply|ns]

  --option normal  (default) renders Gaussians as a point cloud (fast)
  --option ply     renders Gaussians as proper splats via viser's private
                   _add_gaussian_splats API
  --option ns      launches nerfstudio's full Viewer (ns-viewer quality) and
                   overlays the certification cuboids on top
"""

import atexit, os, shutil, sys, json, tempfile, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import viser
from tqdm import tqdm
from scipy.spatial.transform import Rotation as ScipyR

grandfather_path = os.path.abspath(os.path.join(__file__, "../.."))
if grandfather_path not in sys.path:
    sys.path.append(grandfather_path)

from scripts.utils_transform import orthogonal_basis_from_direction


# ─────────────────────────────────────────────────────────────────────────────
# Geometry: partition cell → viser box params
# ─────────────────────────────────────────────────────────────────────────────

def cell_to_world_box(base, direction, radius, xl, xu):
    dir_t            = torch.as_tensor(direction, dtype=torch.float32)
    dir_unit, o1, o2 = orthogonal_basis_from_direction(dir_t)
    dir_unit, o1, o2 = dir_unit.numpy(), o1.numpy(), o2.numpy()

    d_l, xn_l, zn_l = xl
    d_u, xn_u, zn_u = xu
    x_half, z_half   = radius

    center = (base
              + (d_l + d_u) / 2.0 * direction
              + (xn_l + xn_u) / 2.0 * x_half * o1
              + (zn_l + zn_u) / 2.0 * z_half * o2)

    dims = np.array([
        abs(d_u  - d_l)  * np.linalg.norm(direction),
        abs(xn_u - xn_l) * x_half,
        abs(zn_u - zn_l) * z_half,
    ])

    rot_mat = np.column_stack([dir_unit, o1, o2])
    qxyzw   = ScipyR.from_matrix(rot_mat).as_quat()
    wxyz    = np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])
    return center, dims, wxyz


# Unit box vertices (half-extents = 0.5) and face indices — used for
# transparent mesh rendering via add_mesh_simple (which supports opacity).
_BOX_VERTS = np.array([
    [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
    [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
], dtype=np.float32)

_BOX_FACES = np.array([
    [0,2,1],[0,3,2],  # -Z
    [4,5,6],[4,6,7],  # +Z
    [0,1,5],[0,5,4],  # -Y
    [1,2,6],[1,6,5],  # +X
    [2,3,7],[2,7,6],  # +Y
    [3,0,4],[3,4,7],  # -X
], dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Load .pt cells — same filter as test_gatenet_abstract.py
# ─────────────────────────────────────────────────────────────────────────────

def load_cells(abstract_folder):
    pt_files = sorted(f for f in os.listdir(abstract_folder) if f.endswith(".pt"))
    cells, skipped = [], 0
    for fname in pt_files:
        d = torch.load(os.path.join(abstract_folder, fname),
                       map_location="cpu", weights_only=False)
        if d.get("lower_rel") is None or d.get("upper_rel") is None:
            skipped += 1
            continue
        cells.append({
            "point":     np.array(d["point"]),
            "direction": np.array(d["direction"]),
            "radius":    np.array(d["radius"]),
            "xl":        np.array(d["xl"]),
            "xu":        np.array(d["xu"]),
            "lower":     d["lower"],
            "upper":     d["upper"],
            "lower_rel": d["lower_rel"],
            "upper_rel": d["upper_rel"],
        })
    print(f"  {len(cells)} cells loaded  ({skipped} skipped — no lower_rel/upper_rel)")
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# CROWN — mirrors test_gatenet_abstract.py exactly
# ─────────────────────────────────────────────────────────────────────────────

def run_certification(cells, cfg):
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    from DownStreamModel.gatenet.gatenet import GateNet

    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tolerance    = cfg.get("tolerance", 0.02)
    bound_method = cfg.get("bound_method", "backward")
    target_h     = cfg["image_height"]
    target_w     = cfg["image_width"]

    model = GateNet({
        "input_shape":        (3, target_h, target_w),
        "output_shape":       (3,),
        "batch_norm_decay":   cfg["batch_norm_decay"],
        "batch_norm_epsilon": cfg["batch_norm_epsilon"],
    }).to(DEVICE)

    run_dt    = cfg.get("run_datetime")
    ckpt_path = os.path.join(cfg["checkpoint_dir"], run_dt, "latest.pth") if run_dt \
                else os.path.join(cfg["checkpoint_dir"], "latest.pth")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  GateNet: {ckpt_path}")

    colors  = []
    n_green = n_red = 0
    for cell in tqdm(cells, desc="CROWN"):
        lo = cell["lower"].permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        hi = cell["upper"].permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        if lo.shape[-2:] != (target_h, target_w):
            lo = F.interpolate(lo, (target_h, target_w), mode="bilinear", align_corners=False)
            hi = F.interpolate(hi, (target_h, target_w), mode="bilinear", align_corners=False)

        center = (lo + hi) / 2.0
        lirpa  = BoundedModule(model, center,
                               bound_opts={"conv_mode": "patches"}, device=DEVICE)
        ptb    = PerturbationLpNorm(x_L=lo, x_U=hi)
        with torch.no_grad():
            Y_lo, Y_hi = lirpa.compute_bounds(
                x=(BoundedTensor(center, ptb),), method=bound_method)

        Y_lo = Y_lo.squeeze(0).cpu()
        Y_hi = Y_hi.squeeze(0).cpu()
        X_lo = torch.tensor(cell["lower_rel"])
        X_hi = torch.tensor(cell["upper_rel"])

        certified = bool(torch.all(Y_lo >= X_lo - tolerance) and
                         torch.all(Y_hi <= X_hi + tolerance))
        colors.append("green" if certified else "red")
        if certified: n_green += 1
        else:         n_red   += 1

    print(f"  Certified {n_green}/{len(cells)}   violated {n_red}/{len(cells)}")
    return colors


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
#
# .pt files store point/direction in traj.json world space [x, y, z].
# Gaussians (after inverse dataparser_transform) live in nerfstudio
# pre-transform space.  In pose_to_matrix the axis flips remap:
#   traj-world [x, y, z]  →  ns-pre-transform [x, z, -y]
# Apply the same remapping so cuboids and Gaussians share one frame.
# ─────────────────────────────────────────────────────────────────────────────

# Row-vector form: v_display = v_traj @ M_FLIP
M_FLIP = np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]], dtype=np.float64)

# Column-vector form of the same rotation (used for covariance transforms)
# A_FLIP_COL = M_FLIP.T  →  p_display = A_FLIP_COL @ p_traj
A_FLIP_COL = M_FLIP.T  # [[1,0,0],[0,0,1],[0,-1,0]]

# nerfstudio multiplies all scene positions by this when displaying in viser
VISER_NS_SCALE = 10.0


def traj_to_ns(v):
    """Convert a 3-vector from traj.json world space to viser display space
    (ns-pre-transform, used by --option normal/ply)."""
    return np.asarray(v, dtype=np.float64) @ M_FLIP


def load_dataparser_transform(scene_folder):
    """Load R_dp, t_dp, scale_dp from dataparser_transforms.json."""
    tf_path = os.path.join(scene_folder, "dataparser_transforms.json")
    if not os.path.exists(tf_path):
        return np.eye(3), np.zeros(3), 1.0
    with open(tf_path) as f:
        tf = json.load(f)
    T = np.array(tf["transform"], dtype=np.float64)
    return T[:3, :3], T[:3, 3], float(tf["scale"])


def traj_to_ns_viewer(p, R_dp, t_dp, scale_dp, is_vector=False):
    """Convert from traj.json world space to nerfstudio viewer space
    (ns-normalized × VISER_NS_SCALE, used by --option ns).
    is_vector=True skips the translation component (for direction vectors)."""
    p_pre = np.asarray(p, dtype=np.float64) @ M_FLIP   # axis flip
    if is_vector:
        return (p_pre @ R_dp.T) * scale_dp * VISER_NS_SCALE
    return (p_pre @ R_dp.T + t_dp) * scale_dp * VISER_NS_SCALE


# ─────────────────────────────────────────────────────────────────────────────
# Option 'ns': launch nerfstudio's own Viewer, return its viser server
# ─────────────────────────────────────────────────────────────────────────────

def _make_filtered_data_dir(data_dir):
    """Return a Path to a temp directory whose transforms.json only lists frames
    that actually exist on disk.  Symlinks every other item in data_dir so
    nerfstudio finds all other metadata files.  The temp dir is deleted at exit.
    Returns (filtered_path, tmpdir_str) or (data_dir, None) if nothing is missing.
    """
    from pathlib import Path
    data_dir = Path(data_dir)
    tf_path = data_dir / "transforms.json"
    if not tf_path.exists():
        return data_dir, None

    with open(tf_path) as f:
        transforms = json.load(f)

    frames = transforms.get("frames", [])
    existing = [fr for fr in frames if (data_dir / fr["file_path"]).exists()]
    if len(existing) == len(frames):
        return data_dir, None  # nothing missing

    print(f"  WARNING: {len(frames)-len(existing)}/{len(frames)} frames missing — "
          f"using {len(existing)} existing frames for viewer camera display.")

    tmpdir = tempfile.mkdtemp(prefix="ns_filtered_data_")
    # Write filtered transforms.json
    filtered = dict(transforms)
    filtered["frames"] = existing
    with open(os.path.join(tmpdir, "transforms.json"), "w") as f:
        json.dump(filtered, f, indent=2)
    # Symlink everything else (images/, other json files, etc.)
    for item in data_dir.iterdir():
        if item.name == "transforms.json":
            continue
        dst = os.path.join(tmpdir, item.name)
        if not os.path.exists(dst):
            os.symlink(item.resolve(), dst)

    atexit.register(shutil.rmtree, tmpdir, True)
    return Path(tmpdir), tmpdir


def start_ns_viewer(scene_folder, port=8080, data_override=None):
    """Load nerfstudio pipeline and start its Viewer.
    Returns (viser_server, R_dp, t_dp, scale_dp).
    """
    from pathlib import Path
    from threading import Lock

    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.viewer.viewer import Viewer, VISER_NERFSTUDIO_SCALE_RATIO

    ns_config = Path(scene_folder) / "config.yml"
    if not ns_config.exists():
        raise FileNotFoundError(f"nerfstudio config.yml not found: {ns_config}")

    print("  eval_setup: loading nerfstudio pipeline …")

    def _fix_data_path(cfg):
        # Resolve data path
        if data_override is not None:
            raw_data = Path(data_override).resolve()
        else:
            rel = cfg.pipeline.datamanager.data
            if rel is not None and not Path(rel).is_absolute():
                raw_data = Path(os.getcwd()) / rel
            else:
                raw_data = Path(rel) if rel else None

        # Filter out missing frames so nerfstudio doesn't crash on absent images
        if raw_data is not None:
            filtered_data, _ = _make_filtered_data_dir(raw_data)
        else:
            filtered_data = raw_data

        cfg.pipeline.datamanager.data = filtered_data
        cfg.pipeline.datamanager.dataparser.data = filtered_data

        # Fix checkpoint dir.  eval_setup calls config.get_checkpoint_dir()
        # *after* this callback, which reconstructs the path as:
        #   output_dir / experiment_name / method_name / timestamp / relative_model_dir
        # The stale config.yml has output_dir="outputs" and experiment_name="uturn",
        # so we correct both to match the actual scene_folder on disk.
        sp = Path(scene_folder).resolve()
        cfg.output_dir    = sp.parent.parent.parent   # …/nerfstudio/outputs
        cfg.experiment_name = sp.parent.parent.name   # e.g. "train_data_new"
        return cfg

    config, pipeline, _, step = eval_setup(
        ns_config, test_mode="inference",
        update_config_callback=_fix_data_path,
    )

    # Override the viewer port before creating the Viewer
    config.viewer.websocket_port = port
    config.viewer.websocket_port_default_from_train = False

    viewer_log = config.get_base_dir() / config.viewer.relative_log_filename

    viewer = Viewer(
        config.viewer,
        log_filename=viewer_log,
        datapath=pipeline.datamanager.get_datapath(),
        pipeline=pipeline,
        train_lock=Lock(),
        share=False,
    )

    # train_dataset may be None in inference mode — init_scene still works,
    # it just won't show the training-camera frustums in the viewer.
    train_ds = getattr(pipeline.datamanager, "train_dataset", None)
    eval_ds  = getattr(pipeline.datamanager, "eval_dataset",  None)
    viewer.init_scene(
        train_dataset=train_ds,
        train_state="completed",
        eval_dataset=eval_ds,
    )
    viewer.update_scene(step=step)

    R_dp, t_dp, scale_dp = load_dataparser_transform(scene_folder)
    return viewer.viser_server, R_dp, t_dp, scale_dp


# ─────────────────────────────────────────────────────────────────────────────
# Option 'normal': Gaussian means as a point cloud
# ─────────────────────────────────────────────────────────────────────────────

def load_gaussian_means(ckpt_path, scene_folder, max_points=200_000):
    d     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pipe  = d["pipeline"]
    means = pipe["_model.gauss_params.means"].numpy()
    feats = torch.sigmoid(pipe["_model.gauss_params.features_dc"]).numpy()
    opacs = torch.sigmoid(pipe["_model.gauss_params.opacities"]).squeeze(-1).numpy()

    # Inverse dataparser_transform: p_world = (p_ns / scale - t) @ R
    tf_path = os.path.join(scene_folder, "dataparser_transforms.json")
    if os.path.exists(tf_path):
        with open(tf_path) as f:
            tf = json.load(f)
        T     = np.array(tf["transform"])   # 3×4
        R, t  = T[:3, :3], T[:3, 3]
        scale = float(tf["scale"])
        means = (means / scale - t) @ R
    else:
        print(f"  WARNING: dataparser_transforms.json not found at {tf_path}")

    idx = np.argsort(opacs)[::-1][:max_points]
    return means[idx], np.clip(feats[idx] * 255, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Option 'ply': Gaussian splats with proper covariance matrices
#
# viser._add_gaussian_splats(name, centers, covariances, rgbs, opacities)
#   centers     (N,3)   float32  — Gaussian centres in display space
#   covariances (N,3,3) float32  — second-moment matrices in display space
#   rgbs        (N,3)   float32  — colours in [0,1]
#   opacities   (N,1)   float32  — opacity in [0,1]
#
# Coordinate chain (all column-vector convention for covariances):
#   ns-normalised  →(inverse dp-transform)→  ns-pre-transform
#                  →(axis-flip A_FLIP_COL)→  display space
# ─────────────────────────────────────────────────────────────────────────────

def load_gaussian_splats(ckpt_path, scene_folder, max_points=200_000):
    d    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pipe = d["pipeline"]

    means      = pipe["_model.gauss_params.means"].numpy()           # (N,3) ns-norm
    quats      = pipe["_model.gauss_params.quats"].numpy()           # (N,4) wxyz
    log_scales = pipe["_model.gauss_params.scales"].numpy()          # (N,3)
    feats      = torch.sigmoid(pipe["_model.gauss_params.features_dc"]).numpy()  # (N,3)
    opacs      = torch.sigmoid(pipe["_model.gauss_params.opacities"]).numpy()    # (N,1)

    # ── filter top-N by opacity ───────────────────────────────────────────────
    idx = np.argsort(opacs.squeeze(-1))[::-1][:max_points]
    means      = means[idx]
    quats      = quats[idx]
    log_scales = log_scales[idx]
    feats      = feats[idx]
    opacs      = opacs[idx]

    # ── load dataparser transform ─────────────────────────────────────────────
    R_dp   = np.eye(3,  dtype=np.float64)
    t_dp   = np.zeros(3, dtype=np.float64)
    sc_dp  = 1.0
    tf_path = os.path.join(scene_folder, "dataparser_transforms.json")
    if os.path.exists(tf_path):
        with open(tf_path) as f:
            tf = json.load(f)
        T    = np.array(tf["transform"], dtype=np.float64)   # 3×4
        R_dp = T[:3, :3]
        t_dp = T[:3,  3]
        sc_dp = float(tf["scale"])
    else:
        print(f"  WARNING: dataparser_transforms.json not found at {tf_path}")

    # ── means: ns-norm → pre-transform world → display ───────────────────────
    means_world   = (means.astype(np.float64) / sc_dp - t_dp) @ R_dp   # row-vec
    centers_disp  = (means_world @ M_FLIP).astype(np.float32)

    # ── covariances in ns-normalised space ───────────────────────────────────
    # Convert quats from wxyz (nerfstudio) to xyzw (scipy)
    quats_xyzw = np.concatenate([quats[:, 1:], quats[:, :1]], axis=1)
    R_gauss    = ScipyR.from_quat(quats_xyzw).as_matrix()             # (N,3,3)
    s          = np.exp(log_scales.astype(np.float64))                 # (N,3)
    # RS[n] = R_gauss[n] @ diag(s[n])  →  Σ_ns[n] = RS[n] @ RS[n].T
    RS         = R_gauss * s[:, None, :]                               # (N,3,3)
    Sigma_ns   = RS @ RS.transpose(0, 2, 1)                           # (N,3,3)

    # ── Σ_ns → pre-transform world: Σ_world = R_dp.T @ Σ_ns @ R_dp / sc_dp² ─
    Sigma_world = np.einsum('ji,njk,kl->nil', R_dp, Sigma_ns, R_dp) / sc_dp**2

    # ── Σ_world → display: Σ_disp = A_FLIP_COL @ Σ_world @ A_FLIP_COL.T ─────
    Sigma_disp = np.einsum('ij,njk,lk->nil', A_FLIP_COL, Sigma_world, A_FLIP_COL)

    # Ensure positive definiteness: add small diagonal regularisation to guard
    # against float32 numerical precision producing tiny negative eigenvalues.
    Sigma_disp += np.eye(3, dtype=np.float64)[None] * 1e-6
    Sigma_disp = Sigma_disp.astype(np.float32)

    print(f"  Covariances computed — max eigval: "
          f"{np.linalg.eigvalsh(Sigma_disp).max():.4f}")

    return centers_disp, Sigma_disp, feats.astype(np.float32), opacs.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="configs/<case_name>/train_certify_config.yml")
    parser.add_argument("--port",   type=int, default=8080)
    parser.add_argument("--data", default=None,
                        help="absolute path to the dataset folder (only needed for --option ns)")
    parser.add_argument("--option", choices=["normal", "ply", "ns"], default="normal",
                        help="normal = point cloud (default); "
                             "ply = proper Gaussian splats via viser private API; "
                             "ns = full nerfstudio Viewer quality")
    parser.add_argument("--no-cuboids", action="store_true",
                        help="just show the Gaussian scene, skip certification and cuboids")
    parser.add_argument("--opacity", type=float, default=0.35,
                        help="cuboid opacity 0=invisible 1=solid (default 0.35)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Scene config sits next to train_certify_config.yml
    scene_cfg_path = os.path.join(os.path.dirname(args.config), "config.yaml")
    scene_cfg = {}
    if os.path.exists(scene_cfg_path):
        with open(scene_cfg_path) as f:
            scene_cfg = yaml.safe_load(f)

    # Gaussian checkpoint path — same formula as abstract_new_gsplat.py
    script_dir = os.path.dirname(os.path.realpath(__file__))
    gs_ckpt = None
    if scene_cfg:
        scene_folder = os.path.join(
            script_dir,
            f"../nerfstudio/outputs/{scene_cfg['case_name']}/"
            f"{scene_cfg['render_method']}/{scene_cfg['data_time']}"
        )
        gs_ckpt = os.path.join(scene_folder, "nerfstudio_models",
                               scene_cfg["checkpoint_filename"])

    # ── Load + certify (skipped when --no-cuboids) ────────────────────────────
    if not args.no_cuboids:
        print(f"Loading cells from: {cfg['abstract_folder']}")
        cells = load_cells(cfg["abstract_folder"])

        print("\nRunning CROWN …")
        cert_colors = run_certification(cells, cfg)
        n_green = cert_colors.count("green")
        n_red   = cert_colors.count("red")

    # ── Viser / ns-viewer ─────────────────────────────────────────────────────
    R_dp = t_dp = scale_dp = None  # only set for ns mode

    if args.option == "ns":
        if not scene_cfg:
            raise RuntimeError("--option ns requires a valid config.yaml alongside train_certify_config.yml")
        print(f"\nStarting nerfstudio Viewer on port {args.port} …")
        server, R_dp, t_dp, scale_dp = start_ns_viewer(scene_folder, port=args.port,
                                                          data_override=args.data)
        print(f"  ns-viewer ready at http://localhost:{args.port}")
    else:
        server = viser.ViserServer(port=args.port)
        print(f"\nViser at http://localhost:{args.port}")

        # ── Gaussian scene (normal / ply modes only) ──────────────────────────
        if gs_ckpt and os.path.exists(gs_ckpt):
            if args.option == "ply":
                print("Loading Gaussian splats (ply mode) …")
                centers, covs, rgbs, opacities = load_gaussian_splats(gs_ckpt, scene_folder)
                server.scene._add_gaussian_splats(
                    "/scene/gaussians",
                    centers=centers,
                    covariances=covs,
                    rgbs=rgbs,
                    opacities=opacities,
                )
                print(f"  {len(centers):,} Gaussians (splat mode)")
            else:
                print("Loading Gaussian means (normal mode) …")
                g_pts, g_rgb = load_gaussian_means(gs_ckpt, scene_folder)
                server.scene.add_point_cloud("/scene/gaussians", points=g_pts,
                                             colors=g_rgb, point_size=0.005)
                print(f"  {len(g_pts):,} Gaussians (point-cloud mode)")

    # ── Certification cuboids ─────────────────────────────────────────────────
    if not args.no_cuboids:
        # ns mode: positions must be in nerfstudio viewer space (ns-normalised × 10)
        # normal/ply modes: positions in ns-pre-transform world space
        if args.option == "ns":
            def _pt(v):  return traj_to_ns_viewer(v, R_dp, t_dp, scale_dp)
            def _dir(v): return traj_to_ns_viewer(v, R_dp, t_dp, scale_dp, is_vector=True)
            def _rad(r): return np.array(r) * scale_dp * VISER_NS_SCALE
        else:
            def _pt(v):  return traj_to_ns(v)
            def _dir(v): return traj_to_ns(v)
            def _rad(r): return np.array(r)

        COLOR_MAP = {"green": (50, 200, 50), "red": (220, 50, 50)}
        green_handles, red_handles = [], []

        for i, (cell, col) in enumerate(zip(cells, cert_colors)):
            center, dims, wxyz = cell_to_world_box(
                _pt(cell["point"]), _dir(cell["direction"]),
                _rad(cell["radius"]), cell["xl"], cell["xu"])
            verts = _BOX_VERTS * dims.astype(np.float32)
            h = server.scene.add_mesh_simple(
                name=f"/cells/cell_{i:05d}",
                vertices=verts,
                faces=_BOX_FACES,
                color=COLOR_MAP[col],
                opacity=args.opacity,
                side="double",
                position=tuple(center.tolist()),
                wxyz=tuple(wxyz.tolist()),
            )
            (green_handles if col == "green" else red_handles).append(h)

        print(f"  {n_green} green  |  {n_red} red boxes added")

        # ── GUI ───────────────────────────────────────────────────────────────
        server.gui.add_markdown(
            f"**Certified:** {n_green}/{len(cells)} &nbsp; **Violated:** {n_red}/{len(cells)}")
        show_green = server.gui.add_checkbox("Show certified (green)", initial_value=True)
        show_red   = server.gui.add_checkbox("Show violated  (red)",   initial_value=True)

        @show_green.on_update
        def _(_):
            for h in green_handles: h.visible = show_green.value

        @show_red.on_update
        def _(_):
            for h in red_handles: h.visible = show_red.value

    print("Ready — Ctrl+C to stop")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
