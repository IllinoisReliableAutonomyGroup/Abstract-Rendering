import os
import argparse

def update_image_indices(object_name="airplane_grey", domain_type="yaw"):
    folder_path = f"../Outputs/AbstractImages/{object_name}/{domain_type}"

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    png_files = sorted([f for f in os.listdir(folder_path) if f.startswith("ref_") and f.endswith(".png")])
    for idx, file_name in enumerate(png_files):
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, f"ref_{idx:06d}.png")
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update image indices for reference images.")
    parser.add_argument("--object_name", type=str, default="airplane_grey", help="Name of the object.")
    parser.add_argument("--domain_type", type=str, default="yaw", help="Domain type.")
    args = parser.parse_args()

    update_image_indices(object_name=args.object_name, domain_type=args.domain_type)
