import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from PIL import Image
from PIL import Image, ImageOps

def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    """
    Preprocesses an image for input into a segmentation model.

    Parameters:
    - im (np.ndarray): Input image as a NumPy array.
    - model_input_size (list): Target size for the model input.

    Returns:
    - image (torch.Tensor): Preprocessed image tensor.
    """
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bicubic')
    image = im_tensor / 255.0
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    """
    Post-processes the output from the segmentation model.

    Parameters:
    - result (torch.Tensor): Output tensor from the model.
    - im_size (list): Original image size.

    Returns:
    - im_array (np.ndarray): Processed image array.
    """
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bicubic'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

    
def resize_image_keep_aspect(image: Image.Image, target_size: int, rescale=True):
    """
    Resizes an image to a target size while keeping the aspect ratio.

    Parameters:
    - image (PIL.Image.Image): Input image.
    - target_size (int): Target size for the largest dimension.
    - rescale (bool): Whether to rescale the image.

    Returns:
    - resized_image (PIL.Image.Image): Resized image.
    - reduction_factor (float): Factor by which the image was reduced.
    - new_size (tuple): New image size.
    """
    if not rescale:
        return image, 1, image.size

    original_max_dimension = max(image.size)
    ratio = float(target_size) / original_max_dimension
    new_size = tuple([int(x * ratio) for x in image.size])

    # Determine the interpolation method based on the scaling operation
    if ratio < 1:  # Downscaling
        interpolation = Image.Resampling.BOX
    else:  # Upscaling
        interpolation = Image.Resampling.LANCZOS

    resized_image = image.resize(new_size, interpolation)
    reduction_factor = original_max_dimension / max(new_size)

    return resized_image, reduction_factor, new_size