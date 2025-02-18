import numpy as np
import cv2
import os
from PIL import Image
from scipy import ndimage as ndi
from skimage import segmentation
from utilities import resize_image, process_binary_image
from image_io import create_directories

def watershed_segmentation(img, annotation):
    """
    Performs watershed segmentation on the given image using the provided annotation.

    Parameters:
    - img (np.ndarray): Input image.
    - annotation (np.ndarray): Annotation mask.

    Returns:
    - results (np.ndarray): Segmentation results.
    - gradient_magnitude (PIL.Image.Image): Gradient magnitude image.
    - marker_watershed (np.ndarray): Watershed markers.
    - closed_edges (np.ndarray): Closed edges image.
    - bin_img (PIL.Image.Image): Binary image.
    - rgb_img (PIL.Image.Image): RGB image.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Normalize and blur the grayscale image
    gray_normalized = gray / 255.0
    gray_blurred = cv2.GaussianBlur((gray_normalized * 255).astype(np.uint8), (27, 27), 0)

    # Calculate gradients
    sobelx = cv2.Sobel(gray_blurred, cv2.CV_64F, 1, 0, ksize=9)
    sobely = cv2.Sobel(gray_blurred, cv2.CV_64F, 0, 1, ksize=9)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_magnitude *= 255 / np.max(gradient_magnitude)
    #smooth annotation before using it as a marker for watershed

    marker = segmentation.clear_border(annotation)


    marker = cv2.GaussianBlur(marker, (9, 9), 0)

    marker_internal = ndi.binary_erosion(marker, iterations=20)

    external_a = ndi.binary_dilation(marker, iterations=10)

    external_b = ndi.binary_dilation(marker, iterations=100)

    marker_external = external_b ^ external_a

    marker_watershed = np.zeros(marker_external.shape, dtype=np.uint8)
    marker_watershed += marker_internal.astype(np.uint8) * 128
    marker_watershed += marker_external.astype(np.uint8) * 255

    results = segmentation.watershed(gradient_magnitude, marker_watershed)

    # Apply closing to the edges
    kernel = np.ones((3, 3), np.uint8)
    closed_edges = results

    bin_img = Image.fromarray(results)
    rgb_img = Image.fromarray(img)
    gradient_magnitude = Image.fromarray(gradient_magnitude)

    return results, gradient_magnitude, marker_watershed, closed_edges, bin_img, rgb_img

  
def segment_and_store(annotations, bboxes, rgb_img_path, pad):
    """
    Segments and stores grain images based on annotations and bounding boxes.

    Parameters:
    - annotations (list): List of annotation masks.
    - bboxes (list): List of bounding boxes.
    - rgb_img_path (str): Path to the RGB image.
    - pad (int): Padding value.

    Returns:
    - results (dict): Dictionary containing segmented grain images and metadata.
    """
    img_rgb = Image.open(rgb_img_path)
    img_width, img_height = img_rgb.size
    results = {}

    for i, (annotation, bbox) in enumerate(zip(annotations, bboxes)):
        minr, minc, maxr, maxc = bbox
        minr = max(0, minr - pad)
        minc = max(0, minc - pad)
        maxr = min(img_height, maxr + pad)
        maxc = min(img_width, maxc + pad)

        bbox_width = maxc - minc
        bbox_height = maxr - minr

        # Crop the RGB image using the bounding box
        cropped_rgb = img_rgb.crop((minc, minr, maxc, maxr))

        # Resize the cropped image to 512x512
        if cropped_rgb.size[0] < 512 or cropped_rgb.size[1] < 512:
            cropped_rgb_upscayl = cropped_rgb.resize((512, 512), Image.LANCZOS)
        else:
            cropped_rgb_upscayl = cropped_rgb.resize((512, 512), Image.BOX)

        cropped_rgb_upscayl_array = np.array(cropped_rgb_upscayl)

        bin_ann = np.array(annotation) > 0
        cropped_ann = bin_ann[minr:maxr, minc:maxc]

        # Resize the annotation to 512x512
        if cropped_ann.shape[0] < 512 or cropped_ann.shape[1] < 512:
            # Upscaling
            cropped_ann_upscayl = cv2.resize(cropped_ann.astype(np.uint8), (512, 512), interpolation=cv2.INTER_CUBIC)
        else:
            # Downscaling
            cropped_ann_upscayl = cv2.resize(cropped_ann.astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA)

        # Create the grayscale grain image
        if len(cropped_rgb_upscayl_array.shape) == 3:
            grain_gray = cv2.cvtColor(cropped_rgb_upscayl_array, cv2.COLOR_RGB2GRAY)
        else:
            grain_gray = cropped_rgb_upscayl_array

        labeled_grains, sobel_img, markers, outline, bin_img, rgb_img = watershed_segmentation(cropped_rgb_upscayl_array, cropped_ann_upscayl)

        # Determine the resize method based on the bounding box size
        if bbox_width > 512 or bbox_height > 512:
            resize_method = Image.LANCZOS
        else:
            resize_method = Image.BOX

        bin_img = bin_img.resize((bbox_width, bbox_height), resize_method)
        rgb_img = rgb_img.resize((bbox_width, bbox_height), resize_method)
        sobel_img = sobel_img.resize((bbox_width, bbox_height), resize_method)

        results[i] = {
            'grain_img': rgb_img,
            'grain_bin': bin_img,
            'grain_sobel': sobel_img,
            'bbox': (minc, minr, maxc, maxr),
            'markers': markers,
            'annotation:':cropped_ann_upscayl}

    return results

def segment_and_store(annotations, bboxes, rgb_img_path, pad):
    """
    Segments and stores grain images based on annotations and bounding boxes.

    Parameters:
    - annotations (list): List of annotation masks.
    - bboxes (list): List of bounding boxes.
    - rgb_img_path (str): Path to the RGB image.
    - pad (int): Padding value.

    Returns:
    - results (dict): Dictionary containing segmented grain images and metadata.
    """
    img_rgb = Image.open(rgb_img_path)
    img_width, img_height = img_rgb.size
    results = {}

    for i, (annotation, bbox) in enumerate(zip(annotations, bboxes)):
        minr, minc, maxr, maxc = bbox
        minr = max(0, minr - pad)
        minc = max(0, minc - pad)
        maxr = min(img_height, maxr + pad)
        maxc = min(img_width, maxc + pad)

        bbox_width = maxc - minc
        bbox_height = maxr - minr

        # Crop the RGB image using the bounding box
        cropped_rgb = img_rgb.crop((minc, minr, maxc, maxr))

        # Resize the cropped image to 512x512
        if cropped_rgb.size[0] < 512 or cropped_rgb.size[1] < 512:
            cropped_rgb_upscayl = cropped_rgb.resize((512, 512), Image.LANCZOS)
        else:
            cropped_rgb_upscayl = cropped_rgb.resize((512, 512), Image.BOX)

        cropped_rgb_upscayl_array = np.array(cropped_rgb_upscayl)

        bin_ann = np.array(annotation) > 0
        cropped_ann = bin_ann[minr:maxr, minc:maxc]

        # Resize the annotation to 512x512
        if cropped_ann.shape[0] < 512 or cropped_ann.shape[1] < 512:
            # Upscaling
            cropped_ann_upscayl = cv2.resize(cropped_ann.astype(np.uint8), (512, 512), interpolation=cv2.INTER_CUBIC)
        else:
            # Downscaling
            cropped_ann_upscayl = cv2.resize(cropped_ann.astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA)

        # Create the grayscale grain image
        if len(cropped_rgb_upscayl_array.shape) == 3:
            grain_gray = cv2.cvtColor(cropped_rgb_upscayl_array, cv2.COLOR_RGB2GRAY)
        else:
            grain_gray = cropped_rgb_upscayl_array

        labeled_grains, sobel_img, markers, outline, bin_img, rgb_img = watershed_segmentation(cropped_rgb_upscayl_array, cropped_ann_upscayl)

        # Determine the resize method based on the bounding box size
        if bbox_width > 512 or bbox_height > 512:
            resize_method = Image.LANCZOS
        else:
            resize_method = Image.BOX

        bin_img = bin_img.resize((bbox_width, bbox_height), resize_method)
        rgb_img = rgb_img.resize((bbox_width, bbox_height), resize_method)
        sobel_img = sobel_img.resize((bbox_width, bbox_height), resize_method)

        results[i] = {
            'grain_img': rgb_img,
            'grain_bin': bin_img,
            'grain_sobel': sobel_img,
            'bbox': (minc, minr, maxc, maxr),
            'markers': markers
        }

    return results


def remove_background_for_single_grain(grain_data, background_color='white'):
    """
    Removes the background from a single grain image.

    Parameters:
    - grain_data (dict): Dictionary containing grain image data.
    - background_color (str): Background color ('white' or 'black').

    Returns:
    - grain_no_bg_pil (PIL.Image.Image): Image with background removed.
    """
    grain = grain_data["grain_img"]
    mask = grain_data["grain_bin"]
    grain_np = np.array(grain)
    mask_np = np.array(mask)
    mask_np = cv2.GaussianBlur(mask_np, (3, 3), 0)
    _, mask_np = cv2.threshold(mask_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_np = cv2.bitwise_not(mask_np)
    mask_3ch = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)

    # Determine the interpolation method based on the size change
    interpolation = cv2.INTER_AREA if mask_3ch.shape[0] < grain_np.shape[0] or mask_3ch.shape[1] < grain_np.shape[1] else cv2.INTER_CUBIC

    grain_no_bg = cv2.bitwise_and(grain_np, mask_3ch)
    bg_color = (0, 0, 0) if background_color == 'black' else (255, 255, 255)
    background_image = np.full(grain_np.shape, bg_color, dtype=np.uint8)
    combined_image = np.where(mask_3ch == np.array([0, 0, 0]), background_image, grain_no_bg)
    grain_no_bg_pil = Image.fromarray(combined_image)

    return grain_no_bg_pil

def resize_grains(image_dict, reduction_factor):
    """
    Resizes grain images to a target particle circle diameter (PCD) or enlarges them using reduction_factor.

    Parameters:
    - image_dict (dict): Dictionary containing grain images.
    - reduction_factor (float): Image scaling factor used in preprocessing.
    - PCD_Target (int or bool): Target PCD value or False to use reduction_factor only.

    Returns:
    - image_dict (dict): Updated dictionary with resized images.
    """
    for key, images in image_dict.items():
      
        grain_rgb = images['grain_rgb']
        grain_bin = images['grain_bin']
        
        scale_factor = reduction_factor  # Inverse of reduction_factor to revert scaling

        # Determine the resize method based on the scaling factor
        if scale_factor < 1:
            resize_method = Image.Resampling.BOX
        else:
            resize_method = Image.Resampling.LANCZOS

        images['grain_rs'] = resize_image(grain_rgb, scale_factor, resize_method)
        
        grain_bin_rs = resize_image(grain_bin, scale_factor, resize_method)

        images['grain_bin_rs'] = process_binary_image(grain_bin_rs, scale_factor)

        images['scale_factor'] = scale_factor
        images['reduction_factor'] = reduction_factor

    return image_dict

def grid_grains(all_grains, image_path, rescale, background_color, output_dir):
    """
    Creates grids of grain images (RGB and binary) and saves them.

    Parameters:
    - all_grains (dict): Dictionary containing grain images.
    - image_path (str): Path to the original image.
    - rescale (bool): Whether to use rescaled images.
    - background_color (str): Background color ('white' or 'black').
    - output_dir (str): Directory to save the grid images.

    Returns:
    - grid_rgb_img (PIL.Image.Image): The final RGB grid image.
    - grid_bin_img (PIL.Image.Image): The final binary grid image.
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_subdir = os.path.join(output_dir, basename)

    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    grid_rgb_img_path = os.path.join(output_subdir, f"{basename}_Grid_RGB.png")
    grid_bin_img_path = os.path.join(output_subdir, f"{basename}_Grid_Bin.png")

    grain_rgb_key = "grain_rs"
    grain_bin_key = "grain_bin_rs"

    # Determine the size of each grid cell
    Grid_PX = max(max(grain[grain_rgb_key].size) for grain in all_grains.values())

    num_images = len(all_grains)
    num_rows = max(1, int((num_images - 1) / 6) + 1)
    num_cols = min(num_images, 6)

    # Set background color based on the input parameter
    bg_color = (0, 0, 0) if background_color == 'black' else (255, 255, 255)
    bg_color_bin = (255, 255, 255)

    # Create new blank images for the grids with the specified background color
    grid_rgb_img = Image.new('RGB', (Grid_PX * num_cols, Grid_PX * num_rows), color=bg_color)
    grid_bin_img = Image.new('RGB', (Grid_PX * num_cols, Grid_PX * num_rows), color=bg_color_bin)

    i = 0
    for id, grain_info in all_grains.items():
        grain_rgb = grain_info[grain_rgb_key]
        grain_bin = grain_info[grain_bin_key]
        row = i // num_cols
        col = i % num_cols

        # Calculate the position where the current images should be pasted on the grids
        x_offset = Grid_PX * col
        y_offset = Grid_PX * row

        # Create new images for the grains with the same size as Grid_PX and the specified background color
        grain_rgb_bg = Image.new('RGB', (Grid_PX, Grid_PX), color=bg_color)
        grain_bin_bg = Image.new('RGB', (Grid_PX, Grid_PX), color=bg_color_bin)

        # Calculate the positions to paste the grain images to keep them centered
        grain_rgb_x = (Grid_PX - grain_rgb.width) // 2
        grain_rgb_y = (Grid_PX - grain_rgb.height) // 2
        grain_bin_x = (Grid_PX - grain_bin.width) // 2
        grain_bin_y = (Grid_PX - grain_bin.height) // 2

        grain_rgb_bg.paste(grain_rgb, (grain_rgb_x, grain_rgb_y))
        grain_bin_bg.paste(grain_bin, (grain_bin_x, grain_bin_y))

        # Paste the grain images with background onto the grids
        grid_rgb_img.paste(grain_rgb_bg, (x_offset, y_offset))
        grid_bin_img.paste(grain_bin_bg, (x_offset, y_offset))

        i += 1

    # Fill in any remaining empty cells with the background color
    while i < num_rows * num_cols:
        row = i // num_cols
        col = i % num_cols
        x_offset = Grid_PX * col
        y_offset = Grid_PX * row

        empty_rgb_bg = Image.new('RGB', (Grid_PX, Grid_PX), color=bg_color)
        empty_bin_bg = Image.new('RGB', (Grid_PX, Grid_PX), color=bg_color_bin)

        grid_rgb_img.paste(empty_rgb_bg, (x_offset, y_offset))
        grid_bin_img.paste(empty_bin_bg, (x_offset, y_offset))

        i += 1

    # Save the final grid images
    grid_rgb_img.save(grid_rgb_img_path)
    grid_bin_img.save(grid_bin_img_path)
    print(f"Grid RGB image saved at: {grid_rgb_img_path}")
    print(f"Grid Bin image saved at: {grid_bin_img_path}")

    return grid_rgb_img_path, grid_bin_img_path

def save_grains(image_dict, image_path, output_dir):
    """
    Saves resized grain images to the specified directories.

    Parameters:
    - image_dict (dict): Dictionary containing grain images.
    - image_path (str): Path to the original image.
    - output_dir (str): Directory to save output images.
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(output_dir, image_name)
    raw_dir = os.path.join(save_dir, "Raw")
    bin_dir = os.path.join(save_dir, "Bin")
    create_directories([save_dir, raw_dir, bin_dir])

    for key, images in image_dict.items():
        img_name_bin = f"{image_name}_{key}_bin.png"
        img_name = f"{image_name}_{key}.png"
        images['grain_bin_rs'].save(os.path.join(bin_dir, img_name_bin))
        images['grain_rs'].save(os.path.join(raw_dir, img_name))
        images['grain_id'] = os.path.splitext(img_name)[0]
