# Python script that processes all image files in a specified folder (and its subfolders),
# resizes them to 224x224, and saves the resized images into a separate destination folder
# while maintaining the directory structure.

import os
import sys
import time
from PIL import Image

def resize_and_save_images(src_folder, dest_folder, target_size=(224, 224)):
    """
    Resize all images in a folder and its subfolders to the target size and save them to a destination folder.
    src_folder (str): Path to the source folder containing images.
    dest_folder (str): Path to the destination folder to save resized images.
    target_size (tuple): Target size for the images (default: 224x224).
    """

    # go through all files and subdirectories in the source folder
    for root, _, files in os.walk(src_folder):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Construct the full file path
                src_file_path = os.path.join(root, file)
                # print("src_file_path:", src_file_path)

                # Maintain folder structure in the destination folder
                relative_path = os.path.relpath(root, src_folder)
                dest_dir = os.path.join(dest_folder, relative_path)
                os.makedirs(dest_dir, exist_ok=True)

                # Destination file path
                dest_file_path = os.path.join(dest_dir, file)
                # print("dest_file_path:", dest_file_path)

                try:
                    # Open, resize, and save the image
                    with Image.open(src_file_path) as img:
                        # Get original image dimensions
                        original_shape = img.size  # (width, height)

                        img_resized = img.resize(target_size, Image.LANCZOS)
                        img_resized.save(dest_file_path)

                        print(f"Resized: {file} : [{original_shape[1]}x{original_shape[0]}] -> [{target_size[1]}x{target_size[0]}]")

                except Exception as e:
                    print(f"Error processing {src_file_path}: {e}")


######################################################################################
# Read path of folder containing subfolders, images from command line argument
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Missing folder name! Ensure folder name in quotes.\nUsage: resize_and_save_images \"full/path/to/folder/containing/subfolders/images>\"")
    else:
        source_folder = sys.argv[1]

        # create dest folder name as source folder with _resized suffix.
        parent_dir, source_name = os.path.split(source_folder.rstrip(os.sep))
        # print("parent_dir:", parent_dir)
        # print("source_name:", source_name)

        destination_folder = os.path.join(parent_dir, f"{source_name}_resized")

        print(f"Source Folder: {source_folder}")
        print(f"Destination Folder: {destination_folder}")
        resize_and_save_images(source_folder, destination_folder)

