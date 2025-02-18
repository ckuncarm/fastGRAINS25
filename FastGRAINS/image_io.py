# Import statements as shown earlier
# Importing from scripts
from constants import set_material_name, IOU_THRESHOLD, DEVICE, IMAGE_SIZE, RESCALE, PAD, PCD_Target

import os
import shutil
from google.colab import files
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output
import ipywidgets as widgets
from IPython.display import display
import numpy as np

# Setting material name using interactive form
material_name_widget = widgets.Text(
    value='',
    placeholder='Enter material name, e.g., BBSAND',
    description='Material Name:',
    disabled=False
)

def on_button_click(b):
    global material_name
    with output:
        clear_output()
        material_name = material_name_widget.value
        set_material_name(material_name)  # Set the material name dynamically
        print(f"Material name set to: {material_name}")


def upload_image(material_name, images_dir):
    """
    Uploads an image via the Google Colab file uploader, renames it to include the material name, and saves it to the images directory.

    Parameters:
    - material_name (str): The name of the material.
    - images_dir (str): Directory where images are stored.

    Returns:
    - new_filename (str): The path to the uploaded image.
    """
    # Ensure the directory exists
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    uploaded = files.upload()
    for i, filename in enumerate(uploaded.keys(), start=1):
        extension = os.path.splitext(filename)[1]
        new_filename = os.path.join(images_dir, f"{i}_{material_name}{extension}")
        shutil.move(filename, new_filename)
        return new_filename

def upload_images(material_name, images_dir):
    """
    Uploads multiple images via the Google Colab file uploader, renames them to include the material name, and saves them to the images directory.

    Parameters:
    - material_name (str): The name of the material.
    - images_dir (str): Directory where images are stored.

    Returns:
    - new_filenames (list): List of paths to the uploaded images.
    """
    # Ensure the directory exists
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    uploaded = files.upload()
    new_filenames = []

    # Get the current number of files in the directory to continue numbering
    existing_files = os.listdir(images_dir)
    existing_files_count = len(existing_files)

    for i, filename in enumerate(uploaded.keys(), start=existing_files_count + 1):
        extension = os.path.splitext(filename)[1]
        new_filename = os.path.join(images_dir, f"{i}_{material_name}{extension}")
        shutil.move(filename, new_filename)
        new_filenames.append(new_filename)
        print(f"Image uploaded: {new_filename}")

    return new_filenames


def plot_image(image_path):
    """
    Plots an image using matplotlib.

    Parameters:
    - image_path (str): Path to the image file.
    """
    img = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.close()

# Function to read and convert images to NumPy arrays
def read_image(image_path):

    image = Image.open(image_path)
    return np.array(image)

# Function to plot the images side by side
def plot_grids(grid_rgb_img, grid_bin_img):
    # Create a figure with 1x2 subplots
    grid_rgb_img = read_image(grid_rgb_img)
    grid_bin_img = read_image(grid_bin_img)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the RGB image in the first subplot
    axes[0].imshow(grid_rgb_img)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')  # Hide the axis

    # Plot the binary image in the second subplot
    axes[1].imshow(grid_bin_img, cmap='gray')
    axes[1].set_title('Binary Image')
    axes[1].axis('off')  # Hide the axis

    # Display the plot
    plt.show()


def create_directories(directories):
    """
    Creates directories if they do not already exist.

    Parameters:
    - directories (list): List of directory paths to create.
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)