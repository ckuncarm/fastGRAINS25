import os
import sys
import argparse
import ast
import shutil
import functools
import numpy as np
import torch
from PIL import Image
from skimage.segmentation import clear_border
from skimage.measure import regionprops, label
from ultralytics.nn.tasks import SegmentationModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules.conv import Conv
from fastsam import FastSAM, FastSAMPrompt
from utils.tools import convert_box_xywh_to_xyxy
from constants import DEVICE

# Add known classes to safe globals
torch.serialization.add_safe_globals([SegmentationModel, Sequential, Conv])

# Create a patched version of torch.load that always uses weights_only=False
original_torch_load = torch.load
torch.load = functools.partial(original_torch_load, weights_only=False)

def parse_args(args_list):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./FastSAM.pt", help="Model path")
    parser.add_argument("--img_path", type=str, default="./images/dogs.jpg", help="Path to image file")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--iou", type=float, default=0.9, help="IoU threshold")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="./output/", help="Output path")
    parser.add_argument("--better_quality", action='store_true', help="Better quality")
    parser.add_argument("--withContours", action='store_true', help="Draw contours")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to use")
    return parser.parse_args(args_list)

def load_model(model_path, device):
    torch.serialization.add_safe_globals([SegmentationModel])
    model = FastSAM(model_path)
    model.to(device)
    return model

def main(args):
    model = load_model(args.model_path, args.device)
    input_image = Image.open(args.img_path).convert("RGB")
    everything_results = model(input_image, device=args.device, retina_masks=True, imgsz=args.imgsz, conf=args.conf, iou=args.iou)
    prompt_process = FastSAMPrompt(input_image, everything_results, device=args.device)
    return prompt_process.everything_prompt()

def run_inference(image_path, imgsz, iou, use_gpu=True):
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    args_list = [
        '--model_path', './FastSAM.pt',
        '--img_path', image_path,
        '--imgsz', str(imgsz),
        '--iou', str(iou),
        '--conf', '0.9',
        '--better_quality',
        '--withContours',
        '--device', device
    ]
    args = parse_args(args_list)
    return main(args)

def load_annotations(annotations, min_major_axis_length):
    if isinstance(annotations, torch.Tensor):
        annotations = annotations.cpu().numpy()
    
    annotations = [clear_border(ann) for ann in np.array(annotations)]
    
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
    return [ann[0] for ann in filtered_annotations_with_area]
