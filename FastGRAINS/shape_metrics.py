#@title **6. Auxiliary functions**
#@markdown This cell performs the segmentation and processing of uploaded images.
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from PIL import Image
from loess.loess_1d import loess_1d
from matplotlib.path import Path
from skimage.io import imread
from utilities import resize_image, process_binary_image
from skimage.measure import label, perimeter, regionprops
from skimage.morphology import convex_hull_image
from transformers import AutoModelForImageSegmentation
import feret
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist

def resize_grains_to_PCD(image_dict, reduction_factor, PCD_Target=False):
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
        grain_rgb = images['grain_rs']
        grain_bin = images['grain_bin_rs']

        PCD = images['PCD']
        scale_factor = (PCD_Target/PCD)

        # Determine the resize method based on the scaling factor
        if scale_factor < 1:
            resize_method = Image.Resampling.BOX
        else:
            resize_method = Image.Resampling.LANCZOS

        images['grain_rs_PCD'] = resize_image(grain_rgb, scale_factor, resize_method)
        grain_bin_rs_PCD = resize_image(grain_bin, scale_factor, resize_method)
        images['grain_bin_rs_PCD'] = process_binary_image(grain_bin_rs_PCD, scale_factor)
        images['scale_factor'] = scale_factor
        images['reduction_factor'] = reduction_factor


def euclidean_distance(x1, x2):
    """
    Calculates the Euclidean distance between two sets of points.

    Parameters:
    - x1 (np.ndarray): First set of points, shape (N, 2).
    - x2 (np.ndarray): Second set of points, shape (N, 2).

    Returns:
    - distance (np.ndarray): Array of distances between corresponding points.
    """
    x1 = np.array(x1)
    x2 = np.array(x2)

    if x1.shape[1] != 2:
        x1 = x1.T
    if x2.shape[1] != 2:
        x2 = x2.T

    distance = np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
    return distance

def mycarte2polar(X, Y, center, start):
    """
    Converts Cartesian coordinates to polar coordinates with a specified starting quadrant.

    Parameters:
    - X (np.ndarray): X-coordinates.
    - Y (np.ndarray): Y-coordinates.
    - center (tuple): The center point (x, y) for the transformation.
    - start (int): Starting quadrant (1 or 2).

    Returns:
    - ro (np.ndarray): Radii.
    - phi (np.ndarray): Angles in radians.
    """
    deltaXY = np.subtract(np.column_stack([X, Y]), center)
    deltaX = deltaXY[:, 0]
    deltaY = deltaXY[:, 1]

    ro = np.sqrt(deltaX ** 2 + deltaY ** 2)
    n = len(deltaX)
    phi = np.zeros(n)

    if start == 1:  # Start from the first quadrant
        for i in range(n):
            if deltaX[i] > 0 and deltaY[i] < 0:
                phi[i] = np.arctan(np.abs(deltaY[i] / deltaX[i]))
            elif deltaX[i] == 0 and deltaY[i] < 0:
                phi[i] = np.pi / 2
            elif deltaX[i] < 0 and deltaY[i] < 0:
                phi[i] = np.pi - np.arctan(np.abs(deltaY[i] / deltaX[i]))
            elif deltaX[i] < 0 and deltaY[i] == 0:
                phi[i] = np.pi
            elif deltaX[i] < 0 and deltaY[i] > 0:
                phi[i] = np.arctan(np.abs(deltaY[i] / deltaX[i])) + np.pi
            elif deltaX[i] == 0 and deltaY[i] > 0:
                phi[i] = 3 * np.pi / 2
            elif deltaX[i] > 0 and deltaY[i] > 0:
                phi[i] = 2 * np.pi - np.arctan(np.abs(deltaY[i] / deltaX[i]))
            elif deltaX[i] > 0 and deltaY[i] == 0:
                phi[i] = 0

    elif start == 2:  # Start from the second quadrant
        for i in range(n):
            if deltaX[i] < 0 and deltaY[i] > 0:
                phi[i] = np.arctan(np.abs(deltaY[i] / deltaX[i]))
            elif deltaX[i] == 0 and deltaY[i] > 0:
                phi[i] = np.pi / 2
            elif deltaX[i] > 0 and deltaY[i] > 0:
                phi[i] = np.pi - np.arctan(np.abs(deltaY[i] / deltaX[i]))
            elif deltaX[i] > 0 and deltaY[i] == 0:
                phi[i] = np.pi
            elif deltaX[i] > 0 and deltaY[i] < 0:
                phi[i] = np.arctan(np.abs(deltaY[i] / deltaX[i])) + np.pi
            elif deltaX[i] == 0 and deltaY[i] < 0:
                phi[i] = 3 * np.pi / 2
            elif deltaX[i] < 0 and deltaY[i] < 0:
                phi[i] = 2 * np.pi - np.arctan(np.abs(deltaY[i] / deltaX[i]))
            elif deltaX[i] < 0 and deltaY[i] == 0:
                phi[i] = 0

    if phi[-1] == 0:
        phi[-1] = 2 * np.pi

    return ro, phi
def mypolar2carte(ro, phi, center, start):
    """
    Converts polar coordinates back to Cartesian coordinates.

    Parameters:
    - ro (np.ndarray): Radii.
    - phi (np.ndarray): Angles in radians.
    - center (tuple): The center point (x, y) for the transformation.
    - start (int): Starting quadrant (1 or 2).

    Returns:
    - X (np.ndarray): X-coordinates.
    - Y (np.ndarray): Y-coordinates.
    """
    if start == 1:  # Start from the first quadrant
        X = center[0] + ro * np.cos(phi)
        Y = center[1] + ro * np.sin(phi)
    elif start == 2:  # Start from the second quadrant
        X = center[0] - ro * np.cos(phi)
        Y = center[1] + ro * np.sin(phi)
    return X, Y

def maxlinedev(x, y):
    """
    Calculates the maximum deviation from a straight line connecting the first and last points.

    Parameters:
    - x (np.ndarray): X-coordinates.
    - y (np.ndarray): Y-coordinates.

    Returns:
    - maxdev (float): Maximum deviation.
    - index (int): Index of the point with maximum deviation.
    - D (float): Distance between first and last points.
    - totaldev (float): Sum of squared deviations.
    """
    Npts = len(x)

    if Npts == 1:
        print('Warning: Contour of length 1')
        maxdev = 0
        index = 0
        D = 1
        totaldev = 0
        return maxdev, index, D, totaldev
    elif Npts == 0:
        raise ValueError('Error: Contour of length 0')

    D = np.sqrt((x[0] - x[-1]) ** 2 + (y[0] - y[-1]) ** 2)

    if D > np.finfo(float).eps:
        y_diff = y[0] - y[-1]
        x_diff = x[-1] - x[0]
        C = y[-1] * x[0] - y[0] * x[-1]
        d = np.abs(x * y_diff + y * x_diff + C) / D
    else:
        d = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2)
        D = 1

    maxdev = np.max(d)
    index = np.argmax(d)
    totaldev = np.sum(d ** 2)

    return maxdev, index, D, totaldev

def segment_boundary(X, Y, tolerance, display=False):
    """
    Segments a boundary based on a maximum deviation tolerance.

    Parameters:
    - X (np.ndarray): X-coordinates of the boundary.
    - Y (np.ndarray): Y-coordinates of the boundary.
    - tolerance (float): Maximum allowed deviation.
    - display (bool): Whether to display the segmented boundary.

    Returns:
    - seglist (np.ndarray): Segmented boundary points.
    """
    fst = 0
    lst = len(X) - 1

    seglist = [[X[fst], Y[fst]]]

    while fst < lst:
        m, i, _, _ = maxlinedev(X[fst:lst + 1], Y[fst:lst + 1])

        while m > tolerance:
            lst = i + fst
            m, i, _, _ = maxlinedev(X[fst:lst + 1], Y[fst:lst + 1])

        seglist.append([X[lst], Y[lst]])

        fst = lst
        lst = len(X) - 1

    seglist = np.array(seglist)

    if display:
        plt.figure(figsize=(12, 12))
        plt.plot(seglist[:, 0], seglist[:, 1], 'd-', markersize=7, markerfacecolor='none', markeredgecolor='b', linewidth=0.5, color='b')
        plt.axis('off')
        plt.show()

    return seglist

def concave_convex(seglist, center, display=False):
    """
    Determines concave and convex points in a segmented boundary.

    Parameters:
    - seglist (np.ndarray): Segmented boundary points.
    - center (tuple): Center point (x, y).
    - display (bool): Whether to display the convex points.

    Returns:
    - concave (np.ndarray): Array of concave points.
    - convex (np.ndarray): Array of convex points.
    """
    concave = []
    convex = []
    seglist2 = np.vstack([seglist[-2, :], seglist])

    cx, cy = center

    for i in range(len(seglist2) - 2):
        x1, y1 = seglist2[i, :]
        x2, y2 = seglist2[i + 1, :]
        x3, y3 = seglist2[i + 2, :]

        ab1 = [(y3 - y1) / (x3 - x1 + 1e-10), (x3 * y1 - x1 * y3) / (x3 - x1 + 1e-10)]
        ab2 = [(y2 - cy) / (x2 - cx + 1e-10), (x2 * cy - cx * y2) / (x2 - cx + 1e-10)]

        inset = np.linalg.solve([[1, -ab1[0]], [1, -ab2[0]]], [ab1[1], ab2[1]])[::-1]

        d2Ct = np.linalg.norm([x2 - cx, y2 - cy])
        dInter2Ct = np.linalg.norm([inset[0] - cx, inset[1] - cy])

        if d2Ct <= dInter2Ct:
            concave.append(seglist2[i + 1, :])
        else:
            convex.append(seglist2[i + 1, :])

    if display:
        convex = np.array(convex)
        plt.plot(convex[:, 0], convex[:, 1], 'bo', linewidth=2)
        plt.show()

    return np.array(concave), np.array(convex)

# Function to update the grains dictionary with the computed metrics
def update_grains_with_metrics(grains, return_dict):
    for key in grains.keys():
        if key in return_dict:
            grains[key] = {**grains[key], **return_dict[key]}
        else:
            print(f"Warning: Key {key} not found in return_dict")

def fit_circle(points):
    """
    Fits a circle to a set of points using least squares.

    Parameters:
    - points (np.ndarray): Array of points, shape (N, 2).

    Returns:
    - center (np.ndarray): Center of the fitted circle.
    - radius (float): Radius of the fitted circle.
    """
    A = np.hstack([points, np.ones((points.shape[0], 1))])
    b = np.sum(points ** 2, axis=1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    center = x[:2] / 2
    radius = np.sqrt(x[2] + np.sum(center ** 2))
    return center, radius

def find_min_distance(center, boundary_points):
    """
    Finds the minimum distance from a center point to boundary points.

    Parameters:
    - center (np.ndarray): Center point (x, y).
    - boundary_points (np.ndarray): Array of boundary points.

    Returns:
    - min_distance (float): Minimum distance.
    """
    distances = cdist(boundary_points, [center])
    return np.min(distances)

def is_point_in_polygon(point, polygon):
    """
    Checks if a point is inside a given polygon.

    Parameters:
    - point (tuple): Point coordinates (x, y).
    - polygon (np.ndarray): Array of polygon vertices.

    Returns:
    - inside (bool): True if point is inside the polygon, False otherwise.
    """
    path = Path(polygon)
    return path.contains_point(point)

def fit_small_circles(sz, pixel_list, convex, boundary_points, R, factor, min_points):
    """
    Fits small circles to convex points of the grain boundary.

    Parameters:
    - sz (tuple): Size of the image (height, width).
    - pixel_list (np.ndarray): List of pixel indices belonging to the grain.
    - convex (np.ndarray): Convex points.
    - boundary_points (np.ndarray): Boundary points of the grain.
    - R (float): Maximum radius allowed.
    - factor (float): Factor for minimum distance constraint.
    - min_points (int): Minimum number of points required to fit a circle.

    Returns:
    - z (np.ndarray): Centers of the fitted circles.
    - r (np.ndarray): Radii of the fitted circles.
    - range_list (list): List of index ranges used for fitting.
    """
    z = []
    r = []
    cv = convex
    range_list = []
    fp = 0  # Index of the first point
    lp = len(cv) - 1  # Index of the last point

    while lp >= fp + min_points:
        zc, rc = fit_circle(cv[fp:lp + 1])
        min_dis = np.min(euclidean_distance(boundary_points, np.ones((boundary_points.shape[0], 1)) * zc))

        if lp > fp + min_points and (min_dis < factor * rc or rc >= R or
                                     zc[1] < 1 or zc[1] > sz[0] or
                                     zc[0] < 1 or zc[0] > sz[1] or
                                     (int(round(zc[1])) * sz[1] + int(round(zc[0])) - 1) not in pixel_list):
            lp -= 1
            continue
        elif lp == fp + min_points and (min_dis < factor * rc or rc >= R or
                                        zc[1] < 1 or zc[1] > sz[0] or
                                        zc[0] < 1 or zc[0] > sz[1] or
                                        (int(round(zc[1])) * sz[1] + int(round(zc[0])) - 1) not in pixel_list):
            fp += 1
            lp = len(cv) - 1
            continue

        z.append(zc)
        r.append(rc)
        range_list.append((fp, lp + 1))

        fp = lp + 1
        lp = len(cv) - 1

    return np.array(z), np.array(r), range_list

def compute_corner_circles(sz, pixel_list, convex, boundary_points, R, factor, min_points):
    """
    Computes circles at the corners (convex points) of the grain boundary.

    Parameters:
    - sz (tuple): Size of the image (height, width).
    - pixel_list (np.ndarray): List of pixel indices belonging to the grain.
    - convex (np.ndarray): Convex points.
    - boundary_points (np.ndarray): Boundary points of the grain.
    - R (float): Maximum radius allowed.
    - factor (float): Factor for minimum distance constraint.
    - min_points (int): Minimum number of points required to fit a circle.

    Returns:
    - z (np.ndarray): Centers of the fitted circles.
    - r (np.ndarray): Radii of the fitted circles.
    """
    z1, r1, range1 = fit_small_circles(sz, pixel_list, convex, boundary_points, R, factor, min_points)

    if len(z1) == 0:
        return np.array([]), np.array([])

    z2, r2 = np.array([]), np.array([])
    z3, r3 = np.array([]), np.array([])

    if (range1[0][0] - 1) >= 0 and (range1[-1][1] + 1) <= len(convex) and \
            (range1[-1][1] + len(convex) - range1[0][0] - 2 >= min_points):
        cv = np.vstack([convex[range1[-1][1]:], convex[:range1[0][0]]])
        z2, r2, _ = fit_small_circles(sz, pixel_list, cv, boundary_points, R, factor, min_points)

    elif range1[0][0] <= 1 and range1[-1][1] >= len(convex) - 1:
        cv = np.vstack([convex[range1[-1][1]:], convex[:range1[0][0]]])
        z3, r3, _ = fit_small_circles(sz, pixel_list, cv, boundary_points, R, factor, min_points)
        if len(z1) > 1:
            z1 = z1[1:-1]
            r1 = r1[1:-1]

    z = np.vstack([arr for arr in [z1, z2, z3] if arr.size != 0])
    r = np.hstack([arr for arr in [r1, r2, r3] if arr.size != 0])

    return z, r

def compute_shape_metrics(img_input, span, tol, factor, min_points):
    """
    Computes various shape metrics for a given grain image.

    Parameters:
    - img_input (PIL.Image.Image or str): Input image or path to the image.
    - span (float): Fraction for LOESS smoothing.
    - tol (float): Tolerance for boundary segmentation.
    - factor (float): Factor for fitting circles.
    - min_points (int): Minimum number of points for circle fitting.

    Returns:
    - convex (np.ndarray): Convex points.
    - Roundness (float): Calculated roundness metric.
    - Sa, Sc, Sd, Sp, Swl (float): Shape descriptors.
    - AR (float): Aspect ratio.
    - convexity (float): Convexity metric.
    - z (np.ndarray): Centers of corner circles.
    - r (np.ndarray): Radii of corner circles.
    - MCI (tuple): Maximum circle inscribed (center_x, center_y, radius).
    - MCC (tuple): Minimum circle circumscribed (center_x, center_y, radius).
    - d1, d2, de, dins, dcir, minf, maxf (float): Various diameter measurements.
    """

    # Check if img_input is a PIL Image object
    if isinstance(img_input, Image.Image):
        img = np.array(img_input)
    else:
        # Assume img_input is a file path
        img = imread(img_input)

    # Convert to grayscale if the image has multiple channels
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply Otsu's thresholding
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    BW = np.array(~(img >0), dtype=np.uint8)
    level = 255  # Set the threshold to 0
    BW = img >= level
    BW = ~BW
    BW = np.array(BW, dtype=np.uint8)
    lbl = label(BW)
    cc = regionprops(lbl)[0]
    sz = lbl.shape
    pixel_idx_list = np.ravel_multi_index(cc.coords.T, sz)
    pixel_idx_list = pixel_idx_list.reshape(-1, 1)

    # Compute distance map
    dist_map = cv2.distanceTransform(BW, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)

    # Find contours
    contours, _ = cv2.findContours(BW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]

    # Get the X and Y coordinates of the contour
    raw_X = cnt[:, 0, 0]
    raw_Y = cnt[:, 0, 1]

    # Compute center and radius of the minimum enclosing circle
    center, radius = cv2.minEnclosingCircle(cnt)


    ro, phi = mycarte2polar(raw_X, raw_Y, center, 2)

    # Sort and unique polar coordinates
    sorted_indices = np.argsort(phi)
    sorted_phi = phi[sorted_indices]
    sorted_ro = ro[sorted_indices]
    unique_phi, unique_indices = np.unique(sorted_phi, return_index=True)
    unique_ro = sorted_ro[unique_indices]
    phi = unique_phi.copy()
    ro = unique_ro.copy()

    # Apply LOESS smoothing
    xout, ro_mean, wout = loess_1d(phi, ro, frac=span, degree=2)

    # Convert back to Cartesian coordinates
    X, Y = mypolar2carte(ro_mean, phi, center, 2)
    X = np.append(X, X[0])
    Y = np.append(Y, Y[0])

    cartesian = np.column_stack([X, Y])
    boundary_points = cartesian.copy()

    # Create a binary image with the same shape as the original image
    binary_image = np.zeros_like(BW, dtype=np.uint8)

    # Draw and fill the contour on the binary image using OpenCV
    contour = boundary_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.drawContours(binary_image, [contour], -1, color=1, thickness=cv2.FILLED)

    # Label the binary image
    labeled_image = label(binary_image)

    # Obtain the properties of the labeled regions
    props = regionprops(labeled_image)
    boundary_props = props[0]
    min_axis_length = min(boundary_props.major_axis_length, boundary_props.minor_axis_length)

    # Calculate convex hull
    convex_hull = convex_hull_image(binary_image)
    convex_hull_perimeter = perimeter(convex_hull)

    # Calculate convexity
    convexity = convex_hull_perimeter / boundary_props.perimeter

    # Calculate perimeter sphericity
    particle_area = cv2.contourArea(cnt)
    de = np.sqrt(4 * particle_area / np.pi)
    P_r = cv2.arcLength(cnt, True)
    A = np.pi * (de / 2) ** 2
    Sp = 2 * ((np.pi * A) ** .5) / P_r

    # Calculate Aspect Ratio
    minf = feret.min(binary_image)
    maxf = feret.max(binary_image)
    AR = minf / maxf

    # Compute distance map for the binary image
    dist_map = cv2.distanceTransform(binary_image, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
    R = np.max(dist_map)
    dins = 2*R

    RInd = np.argmax(dist_map)

    cy, cx = np.unravel_index(RInd, BW.shape)

    # Segment boundary and compute concave and convex points
    seglist = segment_boundary(X, Y, tol, 0)
    concave, convex = concave_convex(seglist, [cx, cy], 0)

    # Compute minimum area rectangle and enclosing circle
    cnt = np.array(seglist, dtype=np.float32)
    rect = cv2.minAreaRect(cnt)
    c_cum, r_cum = cv2.minEnclosingCircle(cnt)
    dcir = 2 * r_cum
    PCD = 2 * r_cum
    A_cir = np.pi * r_cum ** 2
    Sd = de / dcir
    Sa = particle_area / A_cir

    box = cv2.boxPoints(rect)
    box = np.intp(box)
    width = rect[1][0]
    height = rect[1][1]
    d1 = max(width, height)
    d2 = min(width, height)
    Swl = d2 / d1
    Sc =dins/dcir

    # Compute corner circles
    z, r = compute_corner_circles(sz, pixel_idx_list, convex, boundary_points, R, factor, min_points)

    if len(z) == 0 or len(r) == 0:
        Roundness = 0
    else:
        Roundness = np.mean(r) / R

    # Zip cx, cy, R as MCI and r_cum, c_cum as MCC
    MCI = (cx, cy, R)
    MCC = (c_cum, r_cum)

    return convex, Roundness, Sa, Sc, Sd, Sp, Swl, AR, convexity, z, r, MCI, MCC, d1, d2, de, dins, dcir, minf,maxf
###############################################################
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, fftfreq
from PIL import Image
from skimage import color

def calculate_nrq(image_path):
    """Calculate normalized roughness (NRq) from particle image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path)
        gray = np.array(image.convert('L'))
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Find largest contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea).squeeze()
        if cnt.ndim != 2:
            return None

        # Calculate PPL (Pixels per Particle Length)
        max_dim = max(np.ptp(cnt[:, 0]), np.ptp(cnt[:, 1]))
        centroid = np.mean(cnt, axis=0)

        # Convert to polar coordinates
        theta, r = convert_to_polar_fixed_angular(cnt, centroid)

        # Automatic cutoff selection (1.2% of max amplitude)
        cutoff_wavelength = auto_cutoff_selection(r)

        # Apply Gaussian regression filter
        r_filtered = gaussian_regression_filter(r, cutoff_wavelength)

        # Calculate roughness parameters
        Rq = np.sqrt(np.mean((r - r_filtered)**2))
        NRq = (Rq / max_dim) * 100

        x = centroid[0] + r_filtered * np.cos(theta)
        y = centroid[1] + r_filtered * np.sin(theta)

        smoothed_contour = np.stack([x, y], axis=1).astype(np.float32)
        smoothed_contour = enforce_ccw(smoothed_contour)

        return {
            'ID': os.path.basename(image_path),
            'PPL': max_dim,
            'Rq': Rq,
            'NRq': NRq,
            'contour': cnt,
            'smoothed': smoothed_contour,
            'image': np.array(image)
        }

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def convert_to_polar_fixed_angular(cnt, centroid, angular_step=0.0017):
    """Convert contour to polar coordinates with fixed angular resolution"""
    dx = cnt[:, 0] - centroid[0]
    dy = cnt[:, 1] - centroid[1]

    theta = np.arctan2(dy, dx)
    theta = np.mod(theta, 2*np.pi)
    sort_idx = np.argsort(theta)

    # Interpolate to regular grid
    theta_uniform = np.arange(0, 2*np.pi, angular_step)
    r_uniform = np.interp(theta_uniform, theta[sort_idx], np.hypot(dx, dy)[sort_idx])

    return theta_uniform, r_uniform

def enforce_ccw(contour):
    area = cv2.contourArea(contour, oriented=True)
    if area < 0:
        return contour[::-1]
    return contour

def auto_cutoff_selection(r_profile, cutoff_percent=1.2):
    """Automatically determines cutoff frequency per Section 3.3"""
    n = len(r_profile)
    fft_vals = np.abs(fft(r_profile - np.mean(r_profile)))
    freqs = fftfreq(n, d=0.0017)  # 0.1° sampling

    # Find cutoff amplitude (1.2% of max per paper)
    max_amp = np.max(fft_vals)
    cutoff_amp = cutoff_percent/100 * max_amp

    # Find lowest frequency exceeding cutoff
    valid_freqs = freqs[(fft_vals > cutoff_amp) & (freqs > 0)]
    if len(valid_freqs) == 0:
        return 0.0

    cutoff_freq = np.min(valid_freqs)
    return 1/cutoff_freq  # Convert to wavelength

def gaussian_regression_filter(r_profile, cutoff_wavelength=35, poly_order=2):
    """Apply Gaussian filter with polynomial detrending"""
    filtered = gaussian_filter1d(r_profile, sigma=cutoff_wavelength)
    x = np.linspace(-1, 1, len(r_profile))
    coeffs = np.polyfit(x, filtered - r_profile, poly_order)
    return filtered - np.polyval(coeffs, x)

def reconstruct_contour(r_filtered, theta, centroid):
    """Reconstruct Cartesian coordinates from filtered polar data"""
    x = centroid[0] + r_filtered * np.cos(theta)
    y = centroid[1] + r_filtered * np.sin(theta)
    return np.stack([x, y], axis=1).astype(np.float32)

def process_folder(input_folder, output_file='results.xlsx'):
    """Process all images in folder and save results"""
    results = []
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))
    axes = axes.flatten()

    # Initialize a list to store Rq values
    rq_values = []

    for i, filename in enumerate(os.listdir(input_folder)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        filepath = os.path.join(input_folder, filename)
        data = calculate_nrq(filepath)

        if data:
            # Add to results
            results.append({
                'ID': data['ID'],
                'PPL': data['PPL'],
                'Rq': data['Rq'],
                'NRq': data['NRq']
            })

            # Store Rq value
            rq_values.append(data['NRq'])

            # Plot contours
            if i < len(axes):
                ax = axes[i]
                ax.imshow(data['image'], cmap='gray')
                ax.plot(data['contour'][:, 0], data['contour'][:, 1], 'grey', lw=2)
                ax.plot(data['smoothed'][:, 0], data['smoothed'][:, 1], 'darkorange', lw=1)
                ax.set_title(f"NRq: {data['NRq']:.2f}%", fontsize=14)  # Increase title font size
                ax.axis('off')  # Remove axes

    # Deactivate excess axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')  # Hide unused subplots

    # Calculate average roughness
    avg_rq = np.mean(rq_values)

    # Add average roughness as a text annotation at the top of the figure
    plt.figtext(0.5, .98, f"Average Normalized Roughness (NRq): {avg_rq:.3f}", ha="center", va = "top", fontsize=18, bbox={"facecolor":"darkorange", "alpha":0.5, "pad":5})

    # Save results and plots
    pd.DataFrame(results).to_excel(output_file, index=False)
    plt.tight_layout(rect=[0, 0, 1 , 0.95])  # Ajusta el margen superior
    plt.savefig('contour_comparison.png')
    plt.close()
    return results

def calculate_shape_metrics(image):
    # image = sorted_grains[1]["grain_bin_rs"]
    # Check if img_input is a PIL Image object
    if isinstance(image, Image.Image):
        img = np.array(image.convert('L'))
    else:
        # Assume image is a file path
        img = imread(image)


    # Convert to grayscale if the image has multiple channels
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gray = np.array(image.convert('L'))

    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # cv2.imshow('image', ~binary)
    # Wait for a key press and close the window
    # cv2.waitKey(5000)

    # cv2.destroyAllWindows()
        
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnt = max(contours, key=cv2.contourArea).squeeze()

    labeled_image = label(binary)

    # Obtain the properties of the labeled regions
    props = regionprops(labeled_image)
    boundary_props = props[0]
    min_axis_length = min(boundary_props.major_axis_length, boundary_props.minor_axis_length)

    # Calculate convex hull
    convex_hull = convex_hull_image(binary)

    # Calculate perimeter sphericity
    particle_area = cv2.contourArea(cnt)
    # Calculate the area of the particle and the convex hull
    particle_area = boundary_props.area  # Area of the particle
    convex_hull_area = np.sum(convex_hull)  # Area of the convex hull

    # Calculate convexity (area-based)
    convexity= particle_area / convex_hull_area

    # print(f"Cx (-): {convexity:.3f}")
    ###########################
    # Calculate Aspect Ratio
    minf = feret.min(binary)
    maxf = feret.max(binary)

    AR = minf / maxf

    # print(f"AR (-): {AR:.3f}")

    # Calculate the perimeter sphericity

    de = np.sqrt(4 * particle_area / np.pi)

    P_r = cv2.arcLength(cnt, True)
    A = np.pi * (de / 2) ** 2
    Sp = 2 * ((np.pi * A) ** .5) / P_r

    # print(f"Sp (-): {Sp:.3f}")

    # Calculate the Width-to-Length ratio

    rect = cv2.minAreaRect(cnt)

    box = np.intp(cv2.boxPoints(rect))
    width = rect[1][0]
    height = rect[1][1]

    d1 = max(width, height)
    d2 = min(width, height)

    Swl = d2 / d1

    # print(f"Swl (-): {Swl:.3f}")

    c_cum, r_cum = cv2.minEnclosingCircle(cnt)

    dcir = 2 * r_cum
    # PCD = 2 * r_cum
    A_cir = np.pi * r_cum ** 2
    Sd = de / dcir
    Sa = particle_area / A_cir

    #Calculate SHape-Angularity Group Indicator
    SAGI = np.abs(5.4*(1-AR)-67.8*(1-convexity)-77.9*(1-Sp))
    # print()
    # print(f"SAGI (-): {SAGI:.1f}")

    return (AR, convexity, Sp,  SAGI, de, maxf, minf, dcir)

def smooth_Vangla(cnt):  

    centroid = np.mean(cnt, axis=0)
    theta, r = convert_to_polar_fixed_angular(cnt, centroid)

    #Compute center and radius of the minimum inscribed circle
    # Automatic cutoff selection (1.2% of max amplitude)
    cutoff_wavelength = auto_cutoff_selection(r)

    # Apply Gaussian regression filter
    r_filtered = gaussian_regression_filter(r, cutoff_wavelength)

    # Calculate roughness parameters
    Rq = np.sqrt(np.mean((r - r_filtered)**2))

    max_dim = max(np.ptp(cnt[:, 0]), np.ptp(cnt[:, 1]))

    NRq = (Rq / max_dim) * 100

    X = centroid[0] + r_filtered * np.cos(theta)
    Y = centroid[1] + r_filtered * np.sin(theta)
    
    X = np.append(X, X[0])
    Y = np.append(Y, Y[0])
    
    max_dim = max(np.ptp(cnt[:,0]), np.ptp(cnt[:,1]))

    return X, Y, max_dim, NRq

def calculate_Roundness(
    image, 
    PCD_Target, 
    dcir, 
    min_points, 
    factor, 
    use_vangla_smoothing=True,  # New parameter to select smoothing method
    zheng_span=0.1  # Span parameter for Zheng's method
):
        # image = sorted_grains[1]["grain_bin_rs"]
    # Check if img_input is a PIL Image object
    if isinstance(image, Image.Image):
        img = np.array(image.convert('L'))
    else:
        # Assume image is a file path
        image = Image.open(image)
        img = np.array(image.convert('L'))

    # Convert to grayscale if the image has multiple channels
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale_factor = (PCD_Target/dcir)

    new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))


    if scale_factor < 1:
        # resize_method = Image.Resampling.BOX
        resize_method = Image.Resampling.LANCZOS

    else:
        resize_method = Image.Resampling.LANCZOS


    grain_resize = image.resize(new_size, resize_method)

    # Apply Gaussian Blur
    gray = np.array(grain_resize.convert('L'))

    # Check if img_input is a PIL Image object
    # img = grain_resize

    img = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(img, 128 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    BW = binary.copy()
        
    lbl = label(BW)
    cc = regionprops(lbl)[0]
    sz = lbl.shape

    pixel_idx_list = np.ravel_multi_index(cc.coords.T, sz)
    pixel_idx_list = pixel_idx_list.reshape(-1, 1)

    dist_map = cv2.distanceTransform(BW, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
    R = np.max(dist_map)
    dins = 2*R
    RInd = np.argmax(dist_map)

    cy, cx = np.unravel_index(RInd, BW.shape)

    contours, _ = cv2.findContours(BW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnt = max(contours, key=cv2.contourArea).squeeze()

    _, r_cum = cv2.minEnclosingCircle(cnt)

    PCD = 2 * r_cum

    tol = 1*(0.02*PCD/100)
    centroid = np.mean(cnt, axis=0)

    # Get the X and Y coordinates of the contour
    raw_X = cnt[:,0]
    raw_Y = cnt[:,1]

    # Compute center and radius of the minimum enclosing circle
    center, radius = cv2.minEnclosingCircle(cnt)
    max_dim = max(np.ptp(cnt[:, 0]), np.ptp(cnt[:, 1]))

    # Apply selected smoothing method
    if use_vangla_smoothing:
        # Vangla's smoothing
        theta, r = convert_to_polar_fixed_angular(cnt,centroid)
        cutoff_wavelength = auto_cutoff_selection(r)
        r_filtered = gaussian_regression_filter(r, cutoff_wavelength)
        Rq = np.sqrt(np.mean((r - r_filtered)**2))
        NRq = (Rq / max_dim) * 100
        X = centroid[0] + r_filtered * np.cos(theta)
        Y = centroid[1] + r_filtered * np.sin(theta)
    else:
        # Zheng's smoothing
        raw_X, raw_Y = cnt[:, 0], cnt[:, 1]
        center, _ = cv2.minEnclosingCircle(cnt)
        ro, phi = mycarte2polar(raw_X, raw_Y, center, 2)
        sorted_indices = np.argsort(phi)
        sorted_phi, sorted_ro = phi[sorted_indices], ro[sorted_indices]
        unique_phi, unique_indices = np.unique(sorted_phi, return_index=True)
        unique_ro = sorted_ro[unique_indices]
        xout, ro_mean, _ = loess_1d(unique_phi, unique_ro, frac=zheng_span, degree=2)
        X, Y = mypolar2carte(ro_mean, unique_phi, center, 2)
        NRq = None  # Not used in Zheng's method

    cartesian = np.column_stack([X, Y])
    boundary_points = cartesian.copy()

    smoothed_contour = np.stack([X, Y], axis=1).astype(np.float32)
    smoothed_contour = enforce_ccw(smoothed_contour)

    # Create a binary image with the same shape as the original image
    binary_image = np.zeros_like(BW, dtype=np.uint8)

    # Draw and fill the contour on the binary image using OpenCV
    contour = boundary_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.drawContours(binary_image, [contour], -1, color=1, thickness=cv2.FILLED)

    seglist = segment_boundary(X, Y, tol, 0)

    concave, convex = concave_convex(seglist, [cx, cy], 0)

    z, r = compute_corner_circles(sz, pixel_idx_list, convex, boundary_points, R, factor, min_points)

    Roundness = np.mean(r) / R
        
    grain_resize_np = np.array(~BW)
    # Create a blank black canvas with the same size as grain_resize
    height, width = grain_resize_np.shape[:2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)  # 3-channel image for color

    # Convert grain_resize to grayscale if it's not already
    if len(grain_resize_np.shape) == 3:  # If the image is RGB
        gray_image = cv2.cvtColor(grain_resize_np, cv2.COLOR_RGB2GRAY)
    else:  # If the image is already grayscale
        gray_image = grain_resize_np

    # Draw the grayscale image on the canvas
    canvas[:, :, 0] = gray_image  # Set all channels to the same grayscale value
    canvas[:, :, 1] = gray_image
    canvas[:, :, 2] = gray_image

    # Draw the smoothed contour (green)
    smoothed_contour_int = smoothed_contour.astype(np.int32)

    cv2.polylines(canvas, [smoothed_contour_int], isClosed=True, color=(0, 252, 124), thickness=3)  # lawngreen

    # Draw the minimum enclosing circle (red)
    cv2.circle(canvas, (int(cx), int(cy)), int(R), color=(0, 0, 255), thickness=2)  # red

    # Draw the center of the minimum enclosing circle (red 'x')
    cv2.drawMarker(canvas, (int(cx), int(cy)), color=(0, 0, 255), markerType= 1, markerSize=12, thickness=1)
    cv2.circle(canvas, (int(cx), int(cy)) , 10, color=(0, 0, 255), thickness=1)  # lawngreen

    # Draw the corner circles (green)
    for center, radius in zip(z, r):
        center_int = (int(center[0]), int(center[1]))
        cv2.circle(canvas, center_int, int(radius), color=(0, 252, 124), thickness=2)  # lawngreen
        cv2.drawMarker(canvas, center_int, color=(255, 255, 255), markerType=1, markerSize = 8, thickness=1)  # white 'x'
        cv2.circle(canvas, center_int, 8, color=(255, 255, 255), thickness=1)  # white 'o' (small circle)

    # Display the final image using cv2_imshow (for Colab)
    # Add Roundness value as text on the image
    roundness_text = f"Roundness: {Roundness:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1  # Font size
    font_color = (255, 255, 255)  # White color
    font_thickness = 2  # Thickness of the text

    # Calculate text size and position
    text_size, _ = cv2.getTextSize(roundness_text, font, font_scale, font_thickness)
    text_width, text_height = text_size

    # Define the center of the image
    center_x = width // 2
    center_y = height // 2

    # Calculate the position of the text box
    text_box_x1 = center_x - (text_width // 2) - 10  # Left edge of the text box
    text_box_y1 = center_y - (text_height // 2) - 10  # Top edge of the text box
    text_box_x2 = center_x + (text_width // 2) + 10  # Right edge of the text box
    text_box_y2 = center_y + (text_height // 2) + 10  # Bottom edge of the text box

    # Create a transparent overlay for the text box
    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (text_box_x1, text_box_y1),
        (text_box_x2, text_box_y2),
        (0, 0, 0),  # Black background
        -1,  # Fill the rectangle
    )

    # Blend the overlay with the canvas (transparency)
    alpha = 0.5  # Transparency factor (0 = fully transparent, 1 = fully opaque)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    # Calculate the position of the text (centered in the text box)
    text_x = center_x - (text_width // 2)
    text_y = center_y + (text_height // 2)

    # Put the text on the canvas
    cv2.putText(canvas, roundness_text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    # Resize the canvas to fit the screen
    scale_percent = 50  # Resize to 50% of the original size (adjust as needed)
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    resized_canvas = cv2.resize(canvas, (new_width, new_height), interpolation=cv2.INTER_AREA)
    new_width = int(width / scale_factor )
    # # new_height = int(height / scale_factor )
    # Display the resized image using cv2.imshow
    window_name = 'Resized Image'
    
    # cv2.imshow(window_name, resized_canvas)
    canvas_rgb = cv2.cvtColor(resized_canvas, cv2.COLOR_BGR2RGB)
    R_img = Image.fromarray(canvas_rgb)

    return Roundness, R_img

def compute_shape_metrics(grain, PCD_Target, min_points, factor, use_vangla_smoothing, span):
    """
    Computes various shape metrics for a given grain image.

    Parameters:
    - img_input (PIL.Image.Image or str): Input image or path to the image.
    - span (float): Fraction for LOESS smoothing.
    - tol (float): Tolerance for boundary segmentation.
    - factor (float): Factor for fitting circles.
    - min_points (int): Minimum number of points for circle fitting.

    Returns:
    - convex (np.ndarray): Convex points.
    - Roundness (float): Calculated roundness metric.
    - Sa, Sc, Sd, Sp, Swl (float): Shape descriptors.
    - AR (float): Aspect ratio.
    - convexity (float): Convexity metric.
    - z (np.ndarray): Centers of corner circles.
    - r (np.ndarray): Radii of corner circles.
    - MCI (tuple): Maximum circle inscribed (center_x, center_y, radius).
    - MCC (tuple): Minimum circle circumscribed (center_x, center_y, radius).
    - d1, d2, de, dins, dcir, minf, maxf (float): Various diameter measurements.
    """
    AR, Cx, Sp,  SAGI, de, dfmax, dfmin, dcir = calculate_shape_metrics(grain)

    Roundness, R_image = calculate_Roundness(grain, PCD_Target, dcir, min_points, factor, use_vangla_smoothing, span)
  

    return AR, Cx, Sp,  SAGI, de, dfmax, dfmin, dcir, Roundness, R_image

def compute_metrics(key, grains, PCD_Target, span, use_vangla_smoothing, factor, min_points,return_dict):
    try:
        grain = grains[key]["grain_bin_rs"]       

        metrics = compute_shape_metrics(grain, PCD_Target, min_points, factor, use_vangla_smoothing, span)

        # AR, Cx, Sp,  SAGI, de, dfmax, dfmin, dcir = calculate_shape_metrics(grain)

        # Unpack the metrics
        (AR, Cx, Sp,  SAGI, de, dfmax, dfmin, dcir, Roundness, R_image) = metrics

        # Update the return_dict with the rescaled metrics and the new mm variables
        return_dict[key] = {
            "Roundness": Roundness,
            "Sp": Sp,
            "AR": AR,
            "Cx": Cx,
            "SAGI": SAGI,
            "de": de,
            "dcir": dcir,
            "minf": dfmin,
            "maxf": dfmax,
            "R_image": R_image  # agrega la imagen al diccionario
        }
        
        
    except Exception as e:
        print(f"Error processing key {key}: {e}")
        
def compute_metrics_for_all_keys(grains, PCD_Target, min_points, factor, use_vangla_smoothing, span):
    """
    Computes shape metrics for all grains and updates the grains dictionary.

    Parameters:
    - grains (dict): Dictionary containing grain data.
    - span (float): Fraction for LOESS smoothing.
    - tol (float): Tolerance for boundary segmentation.
    - factor (float): Factor for fitting circles.
    - min_points (int): Minimum number of points for circle fitting.
    """
    return_dict = {}
    for key in grains.keys():
        compute_metrics(key, grains,PCD_Target, span, use_vangla_smoothing, factor, min_points,return_dict)
    update_grains_with_metrics(grains, return_dict)
# Function to update the grains dictionary with the computed metrics

def update_grains_with_metrics(grains, return_dict):
    for key in grains.keys():
        if key in return_dict:
            grains[key] = {**grains[key], **return_dict[key]}
        else:
            print(f"Warning: Key {key} not found in return_dict")
