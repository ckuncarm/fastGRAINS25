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
    model = FastSAM(model_path)
    model.to(device)
    return model

def main(args):
    model = load_model(args.model_path, args.device)
    input_image = Image.open(args.img_path).convert("RGB")
    everything_results = model(input_image, device=args.device, retina_masks=True, imgsz=args.imgsz, conf=args.conf, iou=args.iou)
    prompt_process = FastSAMPrompt(input_image, everything_results, device=args.device)
    ann = prompt_process.everything_prompt()
    # Eliminamos la generación de la imagen
    return ann  # Ahora solo retornamos las anotaciones

def run_inference(image_path, imgsz, iou, use_gpu=True):
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
    annotations = main(args)  # Ahora solo recibimos las anotaciones
    return annotations  # Retornamos solo las anotaciones

def load_annotations(annotations, min_major_axis_length):
    from skimage.segmentation import clear_border
    from skimage.measure import regionprops, label
    import numpy as np
    import torch

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
