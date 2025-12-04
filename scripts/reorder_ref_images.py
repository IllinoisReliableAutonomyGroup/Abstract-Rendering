import os
import argparse
import re
import shutil

def reorder_ref_images(object_name="airplane_grey", domain_type="x"):
    folder_path = f"../Outputs/AbstractImages/{object_name}/{domain_type}"

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return


    # Find all PNG files with the naming pattern "ref_******.png"
    png_files = [f for f in os.listdir(folder_path) if re.match(r"ref_\d{6}\.png", f)]
    if not png_files:
        print("No ref_******.png files found.")
        return

    # Define the target order: "0,1,10,11,12,...,19,2,20,...99"
    first_set = [0, 1, 12, 23, 34, 45, 56, 67, 78, 89]
    second_set = [i for i in range(2, 100) if i not in first_set]
    target_order = first_set + second_set
    print("Target order:", target_order)

    # Map the target order to the order "0,1,2,3,...99"
    target_to_new_order = {target: idx for idx, target in enumerate(target_order)}
    print("Target to new order mapping:", target_to_new_order)
    # Create a temporary folder to store reordered files
    temp_folder = os.path.join(folder_path, "temp_reorder")
    os.makedirs(temp_folder, exist_ok=True)

    # Copy files to the temporary folder with new names based on the new order
    for file_name in png_files:
        original_index = int(re.search(r"ref_(\d+)\.png", file_name).group(1))
        if original_index in target_to_new_order:
            print((original_index, target_to_new_order[original_index]))
            new_index = target_to_new_order[original_index]
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(temp_folder, f"ref_{new_index:06d}.png")
            shutil.copy(old_path, new_path)

    # Remove all existing ref_******.png files in the original folder
    for file_name in png_files:
        os.remove(os.path.join(folder_path, file_name))

    # Move reordered files back to the original folder
    for file_name in os.listdir(temp_folder):
        shutil.move(os.path.join(temp_folder, file_name), folder_path)

    # Remove the temporary folder
    os.rmdir(temp_folder)

    print("Reordering complete. Files have been renamed to match the new order.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorder ref_******.png files to match the new order.")
    parser.add_argument("--object_name", type=str, default="airplane_grey", help="Name of the object.")
    parser.add_argument("--domain_type", type=str, default="z", help="Domain type.")
    args = parser.parse_args()

    reorder_ref_images(object_name=args.object_name, domain_type=args.domain_type)
