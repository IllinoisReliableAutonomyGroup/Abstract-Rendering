# Abstract-Rendering-Toolkit 

### Authors: Chenxi Ji\* , Yangge Li\* , Xiangru Zhong\* , Huan Zhang, Sayan Mitra

### Updated: Chenxi Ji, chenxij2@illinois.edu, 01/30/2026
This repository provides a user-friendly implementation of **Abstract-Rendering**, which computes the set of images that can be rendered from a set of camera poses under a 3D Gaussian Scene, along with downstream applications such as classification, pose estimation, and object detection.

You can find more resources here:  
- [Paper (NeurIPS 2025)](https://mitras.ece.illinois.edu/research/2025/AbstractRendering_Neurips2025.pdf)  
- [Project Website](https://illinoisreliableautonomygroup.github.io/Abstract-Rendering-Webpage/)  
- [Presentation Slides (NeurIPS 2025)](https://neurips.cc/media/neurips-2025/Slides/119130.pdf)

Follow the steps below to set up the environment, gather scene data, and run the scripts.

---
## Workflow
![](figures/block_backup.png)

---

## Demos

| Train Data New | Boeing 787 — Cuboidal | Boeing 787 — Orbital |
|:---:|:---:|:---:|
| <video src="figures/train_data_new.mp4" controls width="100%"></video> | <video src="figures/cuboidal.mp4" controls width="100%"></video> | <video src="figures/orbital.mp4" controls width="100%"></video> |

---

## Setup

### 0. (Optional) Install Nerfstudio

The scene representation is required to follow the Nerfstudio data format. Therefore, installing Nerfstudio is recommended but not strictly required. You may either follow the installation commands provided in
`
nerfstudio_installation_commands.md
`
or refer to the official [Nerfstudio installation guide](https://docs.nerf.studio/quickstart/installation.html) (note that some steps on the website may be outdated).


### 1. Clone the Abstract-Rendering repository

Download the repository from GitHub and remove the bundled `auto_LiRPA` folder (you will link your own copy in Step 2):

```bash
cd ~
git clone --branch master https://github.com/IllinoisReliableAutonomyGroup/Abstract-Rendering.git
```

### 2. Install auto_LiRPA and Create Symbolic Link
Install the neural network verification library *auto_LiRPA*, and symbolic link it under the Abstract-Rendering dictionary.
```bash
cd ~
git clone --branch master https://github.com/Verified-Intelligence/auto_LiRPA.git
cd ~/Abstract-Rendering
rm -rf auto_LiRPA
ln -s ~/auto_LiRPA/auto_LiRPA auto_LiRPA
```

### 3. Download Scene Data
You may either use your existing Nerfstudio data or download the pre-reconstructed [Nerfstudio scenes](https://drive.google.com/drive/folders/1koY1TL30Bty2x0U6VpszKRgMXk61oTkG?usp=drive_link). First create the output directory:

```bash
cd ~/Abstract-Rendering
mkdir -p nerfstudio/outputs
```

After downloading, unzip the scene archive from your Downloads folder and move it into place. Set `case_name` to match the scene you downloaded (e.g. `train_data_new`):

```bash
export case_name=mini_line

cd ~/Downloads
unzip ${case_name}-*.zip

mv ${case_name} ~/Abstract-Rendering/nerfstudio/outputs/
```

The final directory structure should look like:

```
nerfstudio/outputs/
└── ${case_name}/
    └── ${reconstruction_method}/
        └── ${data_time}/
            ├── config.yml
            ├── dataparser_transforms.json
            └── nerfstudio_models/
                └── step-000XXXXXX.ckpt
```

For example, the U-turn scene used in this repository sits at:

```
nerfstudio/outputs/train_data_new/splatfacto/2025-05-09_151825/
```

Below is a visualization of scene *circle*.
![](figures/scene_circle.png)

### 4. Run via Docker

This repository also includes a Dockerfile that sets up a GPU-enabled environment with CUDA, PyTorch, Nerfstudio, and the other required Python dependencies pre-installed. Using Docker is optional but can make the environment more reproducible and easier to share with others.

**Important: Please complete all prior setup steps (1–3) before using Docker in this step.**

- **Prerequisites**: Complete Steps 1–3 above (clone this repo, install and link your local `auto_LiRPA`, and optionally download scene data), have Docker installed on your machine, and install the NVIDIA Container Toolkit if you want to use a GPU from inside the container.
- **Build the image**: From the root of this repository, build a Docker image using the provided Dockerfile, for example under the name `abstract-rendering:latest`:
  ```bash
  cd ~/Abstract-Rendering
  docker build -t abstract-rendering:latest .
  ```
- **Start a container (mount this repo and your local auto_LiRPA)**: Assuming you followed Step 2 to clone auto_LiRPA into `~/auto_LiRPA` and create the symlink in `~/Abstract-Rendering`, run:
  ```bash
  cd ~/Abstract-Rendering
  docker run --gpus all -it --rm \
    -p 8080:8080 \
    -v "$HOME/Abstract-Rendering":/workspace/Abstract-Rendering \
    -v "$HOME/auto_LiRPA":"$HOME/auto_LiRPA" \
    -v "$HOME/.cache/docker-abstract":/root/.cache \
    abstract-rendering:latest \
    /bin/bash
  ```
  The first `-v` makes your local Abstract-Rendering repository visible at `/workspace/Abstract-Rendering` inside the container. The second `-v` mounts your `~/auto_LiRPA` clone at the same absolute path inside the container so that the `auto_LiRPA` symlink in this repo continues to resolve and the code uses your local auto_LiRPA version. The third `-v` persists the CUDA kernel cache across container restarts — without it, gsplat recompiles CUDA kernels every time you start a new container (2–3 min overhead).
- **Inside the container**: Once the container starts, run
  ```bash
  cd /workspace/Abstract-Rendering
  ```
  and you can follow the commands in the *Examples* section below exactly as written to run the rendering, abstract rendering, and downstream verification scripts from inside the container.

## Examples

**Note**: The default GPU memory is 16GB. If you machine has less, please reduce the value of `gs_batch` in `config.yaml`.

---
### Normal Rendering
You can use the command below to render images from a specified set of waypoints in a given scene (e.g. *circle*):
```bash
cd ~/Abstract-Rendering
export case_name=mini_line
python3 scripts/render_gsplat.py --config configs/${case_name}/config.yaml --odd configs/${case_name}/samples.json
```

The rendered image (`ref_######.png`) will be saved under `~/Abstract-Rendering/Outputs/RenderedIamges/${case_name}/${odd_type}`, for example:

![](figures/ref_img.png)

### Abstract Rendering
You can use the command below to generate abstract images from a specified set of waypoints in a given scene (e.g. *circle*):
```bash
cd ~/Abstract-Rendering
export case_name=mini_line
python3 scripts/abstract_gsplat.py --config configs/${case_name}/config.yaml --odd configs/${case_name}/traj.json
```

The rendered images (`abstract_######.pt`) will be saved under 
`~/Abstract-Rendering/Outputs/AbstractIamges/${case_name}/${odd_type}`, and can be visualized by command, (e.g. *circle*):

```bash
cd ~/Abstract-Rendering
export case_name=mini_line
python3 scripts/vis_absimg.py --config configs/${case_name}/vis_absimg.yaml
```

The visualization of abstract image would be like 
![](figures/abs_img.png)
where the top-left subfigure shows sample concrete image from the pose cell; the bottom-left/right subfigure shows lower/upper bound abstract images; the top-right subfigure shows per-pixel difference between bounds as a heatmap.


### Train Gatenet
```bash
cd ~/Abstract-Rendering
export case_name=mini_line
python3 scripts/train_gatenet.py --config configs/${case_name}/gatenet.yml --samples configs/${case_name}/samples.json
```

### Certify Gatenet
```bash
cd ~/Abstract-Rendering
export case_name=mini_line
python3 scripts/certify_gatenet.py --config configs/${case_name}/gatenet.yml
```

### Visualize Certification Results
```bash
cd ~/Abstract-Rendering
export case_name=mini_line
python3 scripts/plot_gatenet.py --config configs/${case_name}/gatenet.yml --traj configs/${case_name}/traj.yaml
```

The visualization of Gatenet Verification is like:

![](figures/result_circle.png)

where green indicates certified regions; red denotes potential
violations; blue indicates gates.

---

### Visualize Certification in the Nerfstudio Viewer

```bash
cd ~/Abstract-Rendering
export case_name=train_data_new
python3 scripts/visualize_abstract_viser.py \
    --config configs/${case_name}/train_certify_config.yml \
    --option ns \
    --data data/uturn
```

Open `http://localhost:8080` in your browser. **Green** = certified, **red** = violated.

**Useful flags:**

| Flag | Effect |
|---|---|
| `--opacity 0.2` | Make cuboids more transparent (default `0.35`) |
| `--no-cuboids` | Show the scene only, skip CROWN and cuboid overlay |
| `--port 8081` | Change the viewer port if 8080 is already in use |

**Results — Train Data New:**

| ε = 0.05 m | ε = 0.10 m | ε = 0.20 m |
|:---:|:---:|:---:|
| ![tdn-0.05](figures/tdn-0.05.png) | ![tdn-0.1](figures/tdn-0.1.png) | ![tdn-0.2](figures/tdn-0.2.png) |

---

## Boeing 787 — Pose Estimation with LSR Certification

This section covers the full pipeline for the **Boeing 787 Nerfstudio** scene using **Linear Set Representation (LSR)** — a tighter certification method that composes CROWN's per-pixel affine bounds with the abstract renderer's pixel-level affine LSR to produce certified affine pose bounds as a function of the original cuboid perturbation.

Two trajectory types are supported, both stored under `Outputs/AbstractImages/boeing787_nerfstudio/cuboid/`:

```
Outputs/AbstractImages/boeing787_nerfstudio/cuboid/
├── cuboidal/    ← cuboidal trajectory (standard U-turn / approach path)
└── orbital/     ← circular orbital trajectory (all-angle views)
```

> **Important:** Both subdirectories are populated by running the same `abstract_gsplat_pose_estimation.py` script with different trajectory files. The `orbital/` folder is **not** created automatically — after running abstract rendering with the orbital trajectory, manually create the subfolder and move the output `.pt` files there:
> ```bash
> mkdir -p Outputs/AbstractImages/boeing787_nerfstudio/cuboid/orbital
> mv Outputs/AbstractImages/boeing787_nerfstudio/cuboid/abstract_*.pt \
>    Outputs/AbstractImages/boeing787_nerfstudio/cuboid/orbital/
> ```

---

<details>
<summary><b>Show pipeline commands (trajectory, abstract rendering, training, certification)</b></summary>

### 1. Prepare Trajectory Files

Trajectory generation scripts are not included in this repository. Generate the trajectory JSON files using your own tools and place them at:

- **Cuboidal:** `configs/boeing787_nerfstudio/traj.json`
- **Orbital:** `configs/boeing787_nerfstudio/traj_orbital.json`

Each entry in the JSON must include `pose`, `pert_type: "cuboid"`, `cuboidal_halflen`, `lower_rel`, `upper_rel`, and `gate` fields. For the orbital case, ensure `cuboidal_halflen` is non-zero in both dimensions (lateral and vertical) — a zero halflen will produce invalid LSR matrices.

---

### 2. Run Abstract Rendering (LSR mode)

Abstract rendering for the boeing case uses `abstract_gsplat_pose_estimation.py` which additionally computes the per-pixel affine LSR matrices (`lA`, `uA`, `lb`, `ub`) needed for composition with CROWN.

**Cuboidal:**
```bash
cd ~/Abstract-Rendering
export case_name=boeing787_nerfstudio
python3 scripts/abstract_gsplat_pose_estimation.py \
    --config configs/${case_name}/config.yaml \
    --odd configs/${case_name}/traj.json
```
Output goes to: `Outputs/AbstractImages/boeing787_nerfstudio/cuboid/`

**Orbital** (run with the orbital trajectory, then move output manually):
```bash
python3 scripts/abstract_gsplat_pose_estimation.py \
    --config configs/${case_name}/config.yaml \
    --odd configs/${case_name}/traj_orbital.json

mkdir -p Outputs/AbstractImages/boeing787_nerfstudio/cuboid/orbital
mv Outputs/AbstractImages/boeing787_nerfstudio/cuboid/abstract_*.pt \
   Outputs/AbstractImages/boeing787_nerfstudio/cuboid/orbital/
```

---

### 3. Configure `train_certify_config.yml`

Edit `configs/boeing787_nerfstudio/train_certify_config.yml`:

| Parameter | Cuboidal | Orbital |
|---|---|---|
| `abstract_folder` | `Outputs/AbstractImages/boeing787_nerfstudio/cuboid/cuboidal` | `Outputs/AbstractImages/boeing787_nerfstudio/cuboid/orbital` |
| `num_epochs` | e.g. `80` | e.g. `80` |
| `use_lsr` | `true` | `true` |
| `lambda_concrete` | `0.0` | `0.0` |
| `lambda_abstract` | `1.0` | `1.0` |
| `bound_method` | `backward` | `backward` |

To **resume from a pretrained checkpoint** or run **certification only** (no training), set:
```yaml
num_epochs: 0
pretrained_checkpoint: "weights/gatenet/boeing787_nerfstudio/<run_datetime>/final_model.pth"
```
The script will automatically reuse the checkpoint's directory and resume any partial certification from where it left off.

---

### 4. Train + Certify (LSR)

```bash
cd ~/Abstract-Rendering
export case_name=boeing787_nerfstudio
python3 scripts/gatenet_train_certify.py \
    --config configs/${case_name}/train_certify_config.yml
```

This single script:
1. Trains GateNet using the CROWN interval-tightness loss on abstract images
2. Runs LSR certification over all partitions, composing CROWN's per-pixel A matrices with the pixel-level LSR from the abstract renderer
3. Saves `lsr_certification.pt` (affine pose bounds per partition) under `weights/gatenet/boeing787_nerfstudio/<run_datetime>/`

Certification checkpoints are saved every 50 partitions — if the process is killed (GPU OOM), re-running the command resumes automatically.

</details>

---

### 5. Visualize — Cuboidal (Nerfstudio 3D Viewer)

```bash
cd ~/Abstract-Rendering
export case_name=boeing787_nerfstudio
python3 scripts/visualize_abstract_viser.py \
    --config configs/${case_name}/train_certify_config.yml \
    --option ns \
    --data ../Downloads/view_scene_mini/AbstractRenderingDataCollection/boeing787_sampled/boeing787_nerfstudio \
    --lsr weights/gatenet/boeing787_nerfstudio/<run_datetime>/lsr_certification.pt \
    --threshold 0.01
```

Open `http://localhost:8080`. Green boxes = certified within threshold; red = violated.

![Viser Visualization](figures/vis_plane.png)

**Useful flags:**

| Flag | Effect |
|---|---|
| `--threshold 0.01` | Error threshold for green/red coloring (metres) |
| `--opacity 0.2` | Make cuboids more transparent |
| `--no-cuboids` | Show scene only, skip certification overlay |
| `--port 8081` | Change viewer port |

**Results — Boeing 787 Cuboidal:**

| ε = 20 cm | ε = 10 cm | ε = 2 cm | ε = 0.2 cm |
|:---:|:---:|:---:|:---:|
| ![c20](figures/cuboidal-20.png) | ![c10](figures/cuboidal-10.png) | ![c2](figures/cuboidal-2.png) | ![c0.2](figures/cuboidal-0.2.png) |

---

### 6. Visualize — Orbital (2D Certification Plot)

For the orbital case a top-down 2D view is more informative than a 3D cuboid overlay. Each arc of the orbit is coloured green (certified) or red (violated), with the aircraft shown at the centre.

```bash
cd ~/Abstract-Rendering
export case_name=boeing787_nerfstudio
python3 scripts/visualize_abstract_viser.py \
    --config configs/${case_name}/train_certify_config.yml \
    --lsr weights/gatenet/boeing787_nerfstudio/<run_datetime>/lsr_certification.pt \
    --threshold 0.01 \
    --plot2d figures/orbital_certification.png
```

The four figures below show certification results at decreasing error thresholds — **ε = 20 cm, 10 cm, 2 cm, 0.2 cm**:

| ε = 20 cm | ε = 10 cm |
|:---:|:---:|
| ![orbital1](figures/orbital_certification1.png) | ![orbital2](figures/orbital_certification2.png) |

| ε = 2 cm | ε = 0.2 cm |
|:---:|:---:|
| ![orbital3](figures/orbital_certification3.png) | ![orbital4](figures/orbital_certification4.png) |

As the threshold tightens, more arc segments turn red — reflecting the growing difficulty of certifying fine-grained pose accuracy across all orbital viewpoints.

---

<details>
<summary><b>Scripts</b></summary>

`render_gsplat.py`:
- Concrete renderer: given a trained Nerfstudio 3D Gaussian scene and a list of poses, it produces standard RGB images along the trajectory.
- Reads `configs/${case_name}/config.yaml` for parameters set by the user and `configs/${case_name}/traj.json` for the pose information.
- Key parameters in `config.yaml`:
  - Scene selection (`render_method`, `case_name`, `data_time`, `checkpoint_filename`): must match the Nerfstudio output you want to render; use the same format for saving the nerfstudio outputs as mentioned above.
  - Resolution vs speed (`width`, `height`, `fx`, `fy`, `downsampling_ratio`): start with the Nerfstudio training values; if rendering is slow or hits GPU memory limits, increase `downsampling_ratio` to render smaller images while keeping intrinsics consistent.
  - Visible depth range (`min_distance`, `max_distance`): if nearby objects are clipped, reduce `min_distance`; if far background clutter dominates, reduce `max_distance`.
  - Memory/performance (`tile_size_render`, `gs_batch`):These can be adjusted based on available GPUs.
  - Background and saving (`bg_img_path` vs `bg_pure_color`, `save_ref`, `save_filename`): choose between a solid background color or an image, and whether/where to save results.
- Output: for each pose in `traj.json`, renders RGB images to `Outputs/RenderedImages/${case_name}/${odd_type}/`.

`abstract_gsplat.py`:
- Abstract renderer: It takes a *linear set* of perturbed poses around a segment of the trajectory and uses auto_LiRPA to propagate that pose uncertainty through the Gaussian splatting renderer, producing for each pixel a certified lower/upper bound (min/max) over all poses in the set.
- Uses the same `configs/${case_name}/config.yaml` and `configs/${case_name}/traj.json` as `render_gsplat.py`.
- Additional `config.yaml` parameters you will tune:
  - `odd_type`: currently `"cylinder"`, meaning the perturbation set is a cylinder around the nominal path between two waypoints.
  - `tile_size_abstract`: tile size for pixels for the abstract renderer; tuned based on GPU memory.
  - `part`: a triplet describing how finely you partition the cylinder (roughly radial / angular / along‑trajectory). Larger values → more cells, tighter bounds, longer computation; smaller values → fewer cells, looser bounds, quicker computation.
  - `save_bound`: if `true`, saves the lower/upper bound images for each pose cell.
  - `N_samples`: number of concrete samples drawn inside each cell when you want example concrete images in addition to the bounds.
- For each consecutive pair of waypoints in `traj.json`, builds the corresponding cylindrical perturbation set, partitions it according to `part`, and then uses `TransferModel + GsplatRGB` wrapped by auto_LiRPA to compute and save abstract records (with per‑pixel min/max) under `Outputs/AbstractImages/${case_name}/${odd_type}/`.

`render_models.py`:
- The rendering back‑end that both concrete and abstract pipelines rely on:
  - `TransferModel`: a wrapper that holds the current camera rotation and base translation (and, for abstract rendering, also the cylinder direction and radius describing the pose cell). Given either a concrete pose or abstract cylinder parameters, it uses `utils_transform.py` to build a full camera pose matrix and then calls the underlying renderer.
  - `GsplatRGBOrigin`: the concrete renderer used by `render_gsplat.py`. It takes Nerfstudio’s Gaussian parameters (means, scales, opacities, colors), preprocesses them once, and for each pose and image tile projects the Gaussians into that tile and alpha‑blends their colors according to the Gaussian splatting algorithm to produce an RGB tile.
  - `GsplatRGB`: the abstract renderer used by `abstract_gsplat.py`. It implements the same splatting idea as `GsplatRGBOrigin`, but is structured for abstract rendering: for a given pose and tile it (i) crops to only Gaussians that can affect that tile, (ii) splits them into batches controlled by `gs_batch` to fit in memory, and (iii) exposes per‑tile alpha and color tensors that encode each Gaussian’s contribution to each pixel. When `TransferModel(GsplatRGB, ...)` is evaluated under auto_LiRPA with a pose cell as input, these tensors become functions of the abstract input; `utils_alpha_blending.py` then performs interval alpha blending on their LiRPA bounds to obtain per‑pixel lower/upper color bounds over all poses in the perturbation set.

`utils_transform.py`:
- Handles all camera and scene coordinate conversions.
- Builds view matrices from translations and rotations, applies the Nerfstudio world transform and scale, and converts camera‑to‑world transforms into the world‑to‑camera form.
- Also provides the cylindrical pose representation used to describe paths and pose cells in abstract rendering (e.g., mapping abstract cylinder coordinates to concrete translations).

`utils_alpha_blending.py`:
- Implements the volume‑rendering step for Gaussian splats.
- For each gaussian, combines opacity and color contributions for each pixel ray using a cumulative product, and extends the same logic to lower/upper bounds in the abstract setting via interval alpha blending.

**Configuration and new cases**:
- Each case (e.g., `circle`, `line`, `uturn`, `mini_line`) has its own subfolder under `configs/` (such as `configs/circle/`) containing at least:
  - `config.yaml`: shared configuration for `render_gsplat.py` and `abstract_gsplat.py` as described above.
  - `traj.yaml` / `traj.json`: trajectory configuration and generated waypoint/pose file.
  - Optional downstream configs such as `gatenet.yml` and `vis_absimg.yaml`.
- When creating a new case, you should create a new folder under `configs/` (for example `configs/my_case/`) and add a new `config.yaml` and trajectory files there, rather than modifying the existing case folders.

</details>

## Citation

If you use this repository or the Abstract-Rendering toolkit in your work, please consider citing our NeurIPS 2025 splotlight poster:

**BibTeX:**
```bibtex
@inproceedings{ji2025abstractrendering,
  title     = {Abstract Rendering: Certified Rendering Under 3D Semantic Uncertainty},
  author    = {Ji, Chenxi and Li, Yangge and Zhong, Xiangru and Zhang, Huan and Mitra, Sayan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) 2025},
  year      = {2025},
  note      = {Poster},
  url       = {https://mitras.ece.illinois.edu/research/2025/AbstractRendering_Neurips2025.pdf}
}
