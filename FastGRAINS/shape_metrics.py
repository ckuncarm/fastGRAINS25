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

def compute_metrics(key, grains, span, tol, factor, min_points, return_dict):
    try:
        metrics = compute_shape_metrics(grains[key]["grain_bin_rs_PCD"], span, tol, factor, min_points)
        # Unpack the metrics
        (convex, Roundness, Sa, Sc, Sd, Sp, Swl, AR, Cx, z, r, MCI, MCC, d1, d2, de, dins, dcir, minf, maxf) = metrics

        # Rescale the diameters by dividing by the scale factor
        # scale_factor = grains[key]["scale_factor"]
        # d1 *= scale_factor
        # d2 *= scale_factor
        # de *= scale_factor
        # dins *= scale_factor
        # dcir *= scale_factor
        # minf *= scale_factor
        # maxf *= scale_factor

        # Update the return_dict with the rescaled metrics and the new mm variables
        return_dict[key] = {
            "convex": convex, "Roundness": Roundness, "Sa": Sa, "Sc": Sc, "Sd": Sd, "Sp": Sp, "Swl": Swl, "AR": AR, "Cx": Cx,
            "z": z, "r": r, "MCI": MCI, "MCC": MCC, "d1": d1, "d2": d2, "de": de, "dins": dins, "dcir": dcir, "minf": minf, "maxf": maxf,
        }
    except Exception as e:
        print(f"Error processing key {key}: {e}")

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


def compute_metrics_for_all_keys(grains, span, tol, factor, min_points):
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
        compute_metrics(key, grains, span, tol, factor, min_points, return_dict)
    update_grains_with_metrics(grains, return_dict)
