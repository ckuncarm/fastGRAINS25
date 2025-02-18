import torch
import os

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
IOU_THRESHOLD = 0.9
RESCALE = True
IMAGE_SIZE = 1024
PAD = 4
PCD_Target = 1200

# Material name (to be set dynamically)
MATERIAL_NAME = None

def set_material_name(name):
    global MATERIAL_NAME, IMAGES_DIR, OUTPUT_DIR, IOU_THRESHOLD, DEVICE, IMAGE_SIZE, RESCALE, PAD
    MATERIAL_NAME = name
    IMAGES_DIR = os.path.join("./data/input/", MATERIAL_NAME)
    OUTPUT_DIR = os.path.join("./data/output/", MATERIAL_NAME)