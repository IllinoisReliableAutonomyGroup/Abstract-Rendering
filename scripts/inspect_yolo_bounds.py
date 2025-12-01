import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True,
        help="Path to yolo_bounds.pt"
            "(e.g., Outputs/AbstractImages/image_name/round/yolo_bounds.pt)",
    )
    args = parser.parse_args()

    data = torch.load(args.path, map_location="cpu")

    print("\n=== Keys in file ===")
    for k in data.keys():
        print(f"  • {k}")

    print("\n=== Tensor shapes ===")
    for k, v in data.items():
        if torch.is_tensor(v):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {type(v)}")

    print("\n=== Example values (first sample) ===")
    for k, v in data.items():
        if torch.is_tensor(v):
            print(f"{k}[0]:\n{v[0]}\n")

if __name__ == "__main__":
    main()
