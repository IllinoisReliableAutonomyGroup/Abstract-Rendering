import os
import sys
import subprocess

# Directory containing this file: .../Abstract-Rendering/examples/airplane/round
HERE = os.path.abspath(os.path.dirname(__file__))

# Repo root is three levels up: round -> airplane -> examples -> Abstract-Rendering
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))

# Make sure the repo root is on sys.path so imports like auto_LiRPA, DownStreamModel, etc. work
if ROOT not in sys.path:
    sys.path.append(ROOT)

print("Repo root:", ROOT)

# Predefined paths/params for this example
ABSTRACT_DIR = os.path.join(ROOT, "Outputs", "AbstractImages", "airplane_grey", "round")
print("Abstract dir:", ABSTRACT_DIR)

RESNET_CKPT  = os.path.join(ROOT, "DownStreamModel", "cifar10_resnet", "resnet4b.pth")
YOLO_ONNX    = os.path.join(ROOT, "DownStreamModel", "yolo", "TinyYOLO.onnx")
GATENET_CKPT = os.path.join(ROOT, "DownStreamModel", "gatenet", "checkpoint_epoch_100.pth")


def run(cmd, desc: str):
    print(f"\n=== Running: {desc} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"=== Finished: {desc} ===\n")


def run_resnet_verification():
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "verify_nn_from_abstract.py"),
        "resnet",
        "--abstract-dir", ABSTRACT_DIR,
        "--model-name", "resnet4b",
        "--ckpt", RESNET_CKPT,
        "--device", "cuda",   # change to "cpu" if needed
    ]
    run(cmd, "ResNet verification")


def run_yolo_verification():
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "verify_nn_from_abstract.py"),
        "yolo",
        "--abstract-dir", ABSTRACT_DIR,
        "--onnx", YOLO_ONNX,
        "--img-size", "52",      # matches ONNXYOLOModel(input_size=52)
        "--device", "cpu",       # safer on memory
        "--max-samples", "10",   # quick check on subset
    ]
    run(cmd, "YOLO verification")


def run_gatenet_verification():
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "verify_nn_from_abstract.py"),
        "gatenet",
        "--abstract-dir", ABSTRACT_DIR,
        "--ckpt", GATENET_CKPT,
        "--img-size", "64",      # matches GateNet config
        "--device", "cuda",      # or "cpu" if you prefer
    ]
    run(cmd, "GateNet verification")


if __name__ == "__main__":
    # You can comment out any of these if you only want some tasks
    run_resnet_verification()
    run_yolo_verification()
    run_gatenet_verification()

    print("\nAll verification tasks completed.")
