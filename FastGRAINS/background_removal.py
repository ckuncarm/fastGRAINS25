# background_removal.py
import os
from PIL import Image, ImageOps
import numpy as np
import torch
from transformers import AutoModelForImageSegmentation
from preprocessing import preprocess_image, postprocess_image, resize_image_keep_aspect
from constants import DEVICE

def remove_background(image_path, imgsz, output_dir, rescale):
    """
    Removes the background from an image using a pre-trained segmentation model.

    Parameters:
    - image_path (str): Path to the input image.
    - imgsz (int): Image size for processing.
    - output_dir (str): Directory to save output images.
    - rescale (bool): Whether to rescale the image.

    Returns:
    - Paths to the resized image and images with different backgrounds.
    """
    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
    model.to(DEVICE)

    img = Image.open(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_subdir = os.path.join(output_dir, basename)
    os.makedirs(output_subdir, exist_ok=True)

    resized_path = os.path.join(output_subdir, f"{basename}_resized.png")
    white_background_path = os.path.join(output_subdir, f"{basename}_white.png")
    black_background_path = os.path.join(output_subdir, f"{basename}_black.png")
    avg_background_path = os.path.join(output_subdir, f"{basename}_avg_color.png")

    img_rs, reduction_factor, new_size = resize_image_keep_aspect(img, imgsz, rescale)
    img_rs.save(resized_path)

    model_input_size = [1024, 1024]
    orig_im = np.array(img_rs)
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to(DEVICE)

    result = model(image)
    result_image = postprocess_image(result[0][0], orig_im_size)
    pil_im = Image.fromarray(result_image)

    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    no_bg_image.paste(img_rs, mask=pil_im)

    # White background
    white_background = Image.new("RGB", new_size, "white")
    white_background.paste(no_bg_image, mask=pil_im)
    white_background.save(white_background_path)

    # Black background
    black_background = Image.new("RGB", new_size, "black")
    black_background.paste(no_bg_image, mask=pil_im)
    black_background.save(black_background_path)

    # Average color background
    avg_color = np.mean(np.array(img_rs), axis=(0, 1)).astype(int)
    avg_color_tuple = tuple(avg_color)
    avg_background = Image.new("RGB", new_size, avg_color_tuple)
    avg_background.paste(no_bg_image, mask=pil_im)
    avg_background.save(avg_background_path)

    return resized_path, black_background_path, white_background_path, avg_background_path, reduction_factor

def invert_image(image_path, invert=True):
    """
    Optionally inverts an image's colors and saves it back to the same path.

    Parameters:
    - image_path (str): Path to the image file.
    - invert (bool): Whether to invert the image.
    """
    if invert:
        image = Image.open(image_path)
        image = ImageOps.invert(image)
        image.save(image_path)
