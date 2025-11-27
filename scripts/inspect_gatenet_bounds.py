import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to gatenet_bounds.pt file"
            "(e.g., Outputs/AbstractImages/airplane_grey/round/gatenet_bounds.pt)",
    )
    args = parser.parse_args()

    print(f"\nLoading {args.path} ...")
    data = torch.load(args.path, map_location="cpu")

    print("\n=== Keys in file ===")
    for k in data.keys():
        print(f"  • {k}")

    print("\n=== Tensor shapes ===")
    for k, v in data.items():
        if torch.is_tensor(v):
            print(f"{k}: shape = {tuple(v.shape)}")
        else:
            print(f"{k}: type = {type(v)}")

    print("\n=== Example values (first record) ===")
    for k, v in data.items():
        if torch.is_tensor(v):
            print(f"{k}[0]: {v[0].numpy()}\n")

    print("\nDone.")

if __name__ == "__main__":
    main()
