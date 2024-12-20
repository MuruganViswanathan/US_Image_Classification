import os
from PIL import Image


def resize_and_save_images(src_folder, dest_folder, target_size=(224, 224)):
    """
    Resize all images in a folder and its subfolders to the target size and save them to a destination folder.

    Args:
        src_folder (str): Path to the source folder containing images.
        dest_folder (str): Path to the destination folder to save resized images.
        target_size (tuple): Target size for the images (default is 224x224).
    """
    # Walk through all files and subdirectories in the source folder
    for root, _, files in os.walk(src_folder):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                # Construct the full file path
                src_file_path = os.path.join(root, file)

                # Maintain folder structure in the destination folder
                relative_path = os.path.relpath(root, src_folder)
                dest_dir = os.path.join(dest_folder, relative_path)
                os.makedirs(dest_dir, exist_ok=True)

                # Destination file path
                dest_file_path = os.path.join(dest_dir, file)

                try:
                    # Open, resize, and save the image
                    with Image.open(src_file_path) as img:
                        img_resized = img.resize(target_size, Image.ANTIALIAS)
                        img_resized.save(dest_file_path)
                        print(f"Processed and saved: {dest_file_path}")
                except Exception as e:
                    print(f"Error processing {src_file_path}: {e}")


# Example usage:
source_folder = "path/to/source/folder"  # Replace with the path to your source folder
destination_folder = "path/to/destination/folder"  # Replace with the path to your destination folder
resize_and_save_images(source_folder, destination_folder)
