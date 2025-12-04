import os
import torch
import argparse

def main(object_name, domain_type, file_name):
    folder_path = f"../Outputs/AbstractImages/{object_name}/{domain_type}"
    file_path = os.path.join(folder_path, file_name)

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    record = torch.load(file_path)
    print("Keys and values in the record:")
    for key, value in record.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test abstract image file.")
    parser.add_argument("--object_name", type=str, default="airplane_grey", help="Name of the object.")
    parser.add_argument("--domain_type", type=str, default="y", help="Domain type.")
    parser.add_argument("--file_name", type=str, default="abstract_000000.pt", help="Name of the file to test.")
    args = parser.parse_args()

    main(args.object_name, args.domain_type, args.file_name)
