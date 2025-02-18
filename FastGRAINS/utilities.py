
from PIL import Image
import cv2
import numpy as np

def resize_image(image, scale_factor, method):
    """
    Resizes an image by a scale factor using the specified resampling method.

    Parameters:
    - image (PIL.Image.Image): Input image.
    - scale_factor (float): Scaling factor.
    - method (int): Resampling method (e.g., Image.LANCZOS).

    Returns:
    - resized_image (PIL.Image.Image): Resized image.
    """
    new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
    resized_image = image.resize(new_size, method)
    return resized_image

def process_binary_image(image, reduction_factor):
    """
    Processes a binary image by applying blurring and thresholding.

    Parameters:
    - image (PIL.Image.Image): Input binary image.
    - reduction_factor (float): Reduction factor.

    Returns:
    - processed_image (PIL.Image.Image): Processed binary image.
    """
    image = cv2.GaussianBlur(np.array(image), (3, 3), 0)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return Image.fromarray(thresh)
