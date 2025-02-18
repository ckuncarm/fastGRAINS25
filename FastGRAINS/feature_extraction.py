import os
import numpy as np
from skimage.measure import regionprops
import cv2
from image_io import create_directories


def get_bboxes(annotations, pad):
    """
    Computes bounding boxes around segmented objects with optional padding.

    Parameters:
    - annotations (list): List of annotation masks.
    - pad (int): Padding value.

    Returns:
    - bboxes (list): List of bounding boxes.
    """
    bboxes = []
    for ann in annotations:
        label_image = np.where(ann, 1, 0)
        props = regionprops(label_image)
        for prop in props:
            minr, minc, maxr, maxc = prop.bbox
            minr = max(minr - pad, 0)
            minc = max(minc - pad, 0)
            maxr = min(maxr + pad, label_image.shape[0])
            maxc = min(maxc + pad, label_image.shape[1])
            bboxes.append((minr, minc, maxr, maxc))
    return bboxes

def get_centroids(annotations):
    """
    Calculates centroids of segmented objects.

    Parameters:
    - annotations (list): List of annotation masks.

    Returns:
    - centroids (list): List of centroids.
    """
    centroids = []
    for ann in annotations:
        label_image = np.where(ann, 1, 0)
        props = regionprops(label_image)
        for prop in props:
            centroids.append(prop.centroid)
    return centroids

import cv2
import numpy as np

def calculate_grain_metrics(grains, reduction_factor, image_path):
    """
    Calculates metrics like bounding rectangle dimensions and PCD for each grain.

    Parameters:
    - grains (dict): Dictionary containing grain data.
    - reduction_factor (float): Image scaling factor.
    - image_path (str): Path to the original image.
    """
    for id, grain_info in grains.items():
        grain = grain_info['grain_bin_rs']
        grain = np.array(grain).astype(np.uint8)
        grain = cv2.GaussianBlur(grain, (3, 3), 0)
        thresh = cv2.threshold(grain, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        d1 = np.linalg.norm(box[0] - box[1]) 
        d2 = np.linalg.norm(box[1] - box[2])
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        PCD = 2 * radius
        grains[id]['d1'] = d1
        grains[id]['d2'] = d2
        grains[id]['PCD'] = PCD
        grains[id]['x'] = x
        grains[id]['y'] = y
        grains[id]['r'] = radius
        grains[id]['image_path'] = image_path

def save_metrics(grains, d1_list, d2_list, PCD_list, paths_list):
    """
    Saves calculated grain metrics to lists.

    Parameters:
    - grains (dict): Dictionary containing grain data.
    - d1_list (list): List to store d1 values.
    - d2_list (list): List to store d2 values.
    - PCD_list (list): List to store PCD values.
    - paths_list (list): List to store image paths.

    Returns:
    - Updated lists with grain metrics.
    """
    for id, grain_info in grains.items():
        d1_list.append(grain_info.get('d1', None))
        d2_list.append(grain_info.get('d2', None))
        PCD_list.append(grain_info.get('PCD', None))
        paths_list.append(grain_info.get('image_path', None))
    return d1_list, d2_list, PCD_list, paths_list
