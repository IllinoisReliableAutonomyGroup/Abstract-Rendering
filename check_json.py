import os
import json
import base64
import argparse
from PIL import Image
from io import BytesIO
import hashlib

def calculate_hash(image_data):
    """Calculate a hash for the image data to identify unique images."""
    return hashlib.md5(image_data).hexdigest()

def check_json(json_path):
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return

    # Load the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Check if the "images" key exists
    if "images" not in data:
        print("The JSON file does not contain the 'images' key.")
        return

    images = data["images"]
    print(f"Total number of image categories: {len(images)}")

    unique_hashes = {"lower": set(), "upper": set()}  # Track unique hashes for "lower" and "upper"

    for key, image_list in images.items():
        print(f"Category '{key}' contains {len(image_list)} images.")
        if len(image_list) > 0:
            try:
                # Decode the first image to check its datatype and size
                image_data = base64.b64decode(image_list[0])
                image = Image.open(BytesIO(image_data))
                print(f"  Example image from '{key}': Format={image.format}, Size={image.size}, Mode={image.mode}")

                # Calculate unique hashes for "lower" and "upper"
                if key in unique_hashes:
                    for img_base64 in image_list:
                        img_data = base64.b64decode(img_base64)
                        unique_hashes[key].add(calculate_hash(img_data))
            except Exception as e:
                print(f"  Error decoding example image from '{key}': {e}")

    # Print the number of unique images for "lower" and "upper"
    for key in ["lower", "upper"]:
        if key in unique_hashes:
            print(f"Category '{key}' contains {len(unique_hashes[key])} unique images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save interactive visualization for abstract images as JSON.")
    parser.add_argument("--object_name", type=str, default="airplane_grey", help="Name of the object.")
    parser.add_argument("--domain_type", type=str, default="z", help="Domain type.")
    args = parser.parse_args()

    object_name = args.object_name
    domain_type = args.domain_type
    json_path = f"Outputs/Analysis/{object_name}/{domain_type}/interactive_plot.json"
    check_json(json_path)
