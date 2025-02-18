import os
import sys
import torch
import argparse
import ast
import shutil
from PIL import Image
from fastsam import FastSAM, FastSAMPrompt
from utils.tools import convert_box_xywh_to_xyxy
import numpy as np
from constants import DEVICE

def parse_args(args_list):
    """
    Parses command-line-like arguments for model configuration.

    Parameters:
    - args_list (list): List of argument strings.

    Returns:
    - args (Namespace): Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./FastSAM.pt", help="Model path")
    parser.add_argument("--img_path", type=str, default="./images/dogs.jpg", help="Path to image file")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--iou", type=float, default=0.9, help="IoU threshold")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="./output/", help="Output path")
    parser.add_argument("--better_quality", type=bool, default=False, help="Better quality")
    parser.add_argument("--withContours", type=bool, default=False, help="Draw contours")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to use")
    return parser.parse_args(args_list)

def load_model(model_path, device):
    """
    Loads the FastSAM segmentation model onto the specified device.

    Parameters:
    - model_path (str): Path to the model.
    - device (torch.device): Device to load the model on.

    Returns:
    - model (FastSAM): Loaded model.
    """
    model = FastSAM(model_path)
    model.to(device)
    return model

def main(args):
    """
    Main function to execute the segmentation model.

    Parameters:
    - args (Namespace): Parsed arguments.

    Returns:
    - ann (list): Annotations from the model.
    - output_path (str): Path to the output image.
    """
    model = load_model(args.model_path, args.device)
    input_image = Image.open(args.img_path).convert("RGB")
    everything_results = model(input_image, device=args.device, retina_masks=True, imgsz=args.imgsz, conf=args.conf, iou=args.iou)
    prompt_process = FastSAMPrompt(input_image, everything_results, device=args.device)
    ann = prompt_process.everything_prompt()
    output_path = os.path.join(args.output, os.path.basename(args.img_path))
    prompt_process.plot(annotations=ann, output_path=output_path, better_quality=args.better_quality, withContours=args.withContours)
    return ann, output_path

def run_inference(image_path, imgsz, iou, use_gpu=True):
    """
    Sets up arguments and runs the main function to perform inference.

    Parameters:
    - image_path (str): Path to the input image.
    - imgsz (int): Image size for processing.
    - iou (float): IoU threshold.
    - use_gpu (bool): Whether to use GPU.

    Returns:
    - annotations (list): Annotations from the model.
    - output_path (str): Path to the output image.
    """
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    args_list = [
        '--model_path', './FastSAM.pt',
        '--img_path', image_path,
        '--imgsz', str(imgsz),
        '--iou', str(iou),
        '--conf', '0.9',
        '--better_quality', 'True',
        '--withContours', 'True',
        '--device', device
    ]
    args = parse_args(args_list)
    annotations, output_path = main(args)
    return annotations, output_path
def load_annotations(annotations, min_major_axis_length):
    """
    Processes segmentation annotations by filtering small objects and sorting.

    Parameters:
    - annotations (list): List of annotation masks.
    - min_major_axis_length (int): Minimum size to keep an annotation.

    Returns:
    - sorted_filtered_annotations (list): Processed annotations.
    """
    from skimage.segmentation import clear_border
    from skimage.measure import regionprops, label
    import numpy as np
    import torch

    # Convert annotations to numpy arrays if they are PyTorch tensors
    if isinstance(annotations, torch.Tensor):
        annotations = annotations.cpu().numpy()

    annotations = np.array(annotations)
    annotations = [clear_border(ann) for ann in annotations]

    filtered_annotations_with_area = []
    for ann in annotations:
        label_image = label(ann)
        props = regionprops(label_image)
        filtered_ann = np.zeros_like(ann)
        for prop in props:
            if prop.major_axis_length > min_major_axis_length:
                filtered_ann[label_image == prop.label] = 1
        if np.any(filtered_ann):
            area = np.sum(filtered_ann)
            filtered_annotations_with_area.append((filtered_ann, area))

    filtered_annotations_with_area.sort(key=lambda x: x[1], reverse=True)
    sorted_filtered_annotations = [ann[0] for ann in filtered_annotations_with_area]
    return sorted_filtered_annotations