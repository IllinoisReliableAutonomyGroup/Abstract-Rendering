import os
import numpy as np
from PIL import Image

def compute_and_save_abstract_images(save_folder_full):
    # Lists to hold loaded images
    lb_images = []
    ub_images = []
    ref_images = []
    
    # Load images
    for fname in os.listdir(save_folder_full):
        if fname.startswith("lb_") and fname.endswith(".png"):
            img = np.array(Image.open(os.path.join(save_folder_full, fname)), dtype=np.float32) / 255.0
            lb_images.append(img)
        elif fname.startswith("ub_") and fname.endswith(".png"):
            img = np.array(Image.open(os.path.join(save_folder_full, fname)), dtype=np.float32) / 255.0
            ub_images.append(img)
        elif fname.startswith("ref_") and fname.endswith(".png"):
            img = np.array(Image.open(os.path.join(save_folder_full, fname)), dtype=np.float32) / 255.0
            ref_images.append(img)

    # Check if images are found
    if lb_images:
        # Stack and compute pixel-wise minimum (unified lower bound)
        lb_stack = np.stack(lb_images, axis=0)
        unified_lb = np.min(lb_stack, axis=0)
        unified_lb_img = (unified_lb.clip(0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(unified_lb_img).save(os.path.join(save_folder_full, "unified_lb.png"))
        print("Unified lower bound saved as unified_lb.png")
    else:
        print("No lb images found.")

    if ub_images:
        # Stack and compute pixel-wise maximum (unified upper bound)
        ub_stack = np.stack(ub_images, axis=0)
        unified_ub = np.max(ub_stack, axis=0)
        unified_ub_img = (unified_ub.clip(0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(unified_ub_img).save(os.path.join(save_folder_full, "unified_ub.png"))
        print("Unified upper bound saved as unified_ub.png")
    else:
        print("No ub images found.")

    if ref_images:
        # Stack and compute pixel-wise maximum (unified reference)
        ref_stack = np.stack(ref_images, axis=0)

        unified_sam_lb = np.min(ref_stack, axis=0)
        unified_sam_lb_img = (unified_sam_lb.clip(0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(unified_sam_lb_img).save(os.path.join(save_folder_full, "unified_sam_lb.png"))
        print("Unified sampling lb bound saved as unified_sam_lb.png")

        unified_sam_ub = np.max(ref_stack, axis=0)
        unified_sam_ub_img = (unified_sam_ub.clip(0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(unified_sam_ub_img).save(os.path.join(save_folder_full, "unified_sam_ub.png"))
        print("Unified sampling ub bound saved as unified_sam_ub.png")
    else:
        print("No ref images found.")

if __name__ == '__main__':
    # Folder where images are saved
    object_name = "airplane_grey"
    domain_type = "x"

    script_dir = os.path.dirname(os.path.realpath(__file__))
    save_folder_path = "../../Outputs/AbstractImages/"+object_name+"/"+domain_type
    save_folder_full =  os.path.join(script_dir, save_folder_path)
    compute_and_save_abstract_images(save_folder_full)