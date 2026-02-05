# 1) Base image with CUDA + cuDNN for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 2) Basic system tools and Python
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# 3) PyTorch with CUDA 12.1 (adjust if needed)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4) Nerfstudio, auto_LiRPA, and other Python deps used by this repo
RUN pip install "nerfstudio[full]"
RUN pip install "git+https://github.com/Verified-Intelligence/auto_LiRPA.git"

RUN pip install \
    numpy scipy pillow matplotlib tqdm pyyaml torchvision opencv-python

# 5) Copy the Abstract-Rendering repo into the image
WORKDIR /workspace
COPY . /workspace/Abstract-Rendering
WORKDIR /workspace/Abstract-Rendering

# 6) Default command: start a shell; you run scripts manually
CMD ["/bin/bash"]