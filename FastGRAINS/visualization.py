
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from IPython.display import clear_output
from google.colab import files


def plot_grain(grain_data, convex, Roundness, Sa, Sd, Sp, Swl, AR, Cx, z, r, MCI, MCC):
    """
    Plots a single grain with interactive parameters and annotations.

    Parameters:
    - grain_data (dict): Dictionary containing grain image data.
    - convex (np.ndarray): Array of convex points.
    - Roundness, Sa, Sd, Sp, Swl, AR, Cx (float): Shape metrics.
    - z (np.ndarray): Centers of fitted circles.
    - r (np.ndarray): Radii of fitted circles.
    - MCI (tuple): Maximum Circle Inscribed (center_x, center_y, radius).
    - MCC (tuple): Minimum Circle Circumscribed (center_x, center_y, radius).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if 'grain_bin_rs_PCD' not in grain_data:
        print("Error: 'grain_bin_rs_PCD' key is missing in grain_data")
        return

    grain = grain_data['grain_bin_rs_PCD']
    grain_org = grain_data['grain_rs_PCD']

    # Create the plot
    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor('black')
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    # First subplot with grain and annotations
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(grain, cmap='gray_r', aspect='equal')
    ax1.set_facecolor('black')  # Set axis background color to black

    theta = np.linspace(0, 2 * np.pi, 100)
    for center, radius in zip(z, r):
        ax1.plot(center[0], center[1], 'o', markersize=8, markerfacecolor='none',
                 markeredgecolor='w', markeredgewidth=1, label='Circle Center')
        ax1.plot(center[0], center[1], 'x', markersize=6, markerfacecolor='none',
                 markeredgecolor='w', markeredgewidth=1)
        ax1.plot(np.cos(theta) * radius + center[0], np.sin(theta) * radius + center[1],
                 linewidth=1, color="lawngreen", linestyle="-", label='Circles Corners')

    # Plot MCC and MCI
    ax1.plot(np.cos(theta) * MCC[1] + MCC[0][0], np.sin(theta) * MCC[1] + MCC[0][1],
             linewidth=2, color="blue", linestyle="-", label='MCC')
    ax1.plot(np.cos(theta) * MCI[2] + MCI[0], np.sin(theta) * MCI[2] + MCI[1],
             linewidth=2, color="red", linestyle="-", label='MCI')

    # Plot convex points
    ax1.plot(convex[:, 0], convex[:, 1], 'o', markersize=6, markerfacecolor='none',
             markeredgecolor='r', markeredgewidth=0.5, label='Convex Points')

    # Add text annotations
    ax1.text(0.05, 0.95,
             f'Roundness: {Roundness:.2f}\nSa: {Sa:.2f}\nSd: {Sd:.2f}\nSp: {Sp:.2f}\nSwl: {Swl:.2f}\nAR: {AR:.2f}\nCx: {Cx:.2f}',
             transform=ax1.transAxes, fontsize=6, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'),
             fontdict={'style': 'italic'})

    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    ax1.axis("off")

    # Second subplot with raw grain image
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('black')
    ax2.imshow(grain_org, cmap='gray', aspect='equal')
    ax2.plot(np.cos(theta) * MCC[1] + MCC[0][0], np.sin(theta) * MCC[1] + MCC[0][1],
             linewidth=2, color="red", linestyle="-", label='Reference Circle')
    ax2.axis("off")

    plt.show()
    plt.close()

def interactive_plot(id, span, tol, factor, grains, out):
    """
    Interactive plotting function for a single grain, recalculates metrics with given parameters.

    Parameters:
    - id (int): Grain ID.
    - span (float): LOESS smoothing fraction.
    - tol (float): Tolerance for boundary segmentation.
    - factor (float): Factor for circle fitting.
    - grains (dict): Dictionary containing all grains data.
    - out (IPython.display.Output): Output widget for displaying the plot.
    """
    from IPython.display import clear_output
    from shape_metrics import compute_shape_metrics

    grain_data = grains[id]
    try:
        metrics = compute_shape_metrics(grain_data["grain_bin_rs_PCD"], span, tol, factor, min_points=4)
        convex, Roundness, Sa, Sc, Sd, Sp, Swl, AR, Cx, z, r, MCI, MCC, _, _, _, _, _, _, _ = metrics
    except KeyError as e:
        print(f"KeyError: {e} is missing in grain_data")
        return
    with out:
        clear_output(wait=False)
        plot_grain(grain_data, convex, Roundness, Sa, Sd, Sp, Swl, AR, Cx, z, r, MCI, MCC)
        plt.show()
        plt.close("all")

def plot_grains_with_circles(grains, path, num_columns=4):
    """
    Plots all grains with their fitted circles and saves the figure and data.

    Parameters:
    - grains (dict): Dictionary containing all grains data.
    - path (str): Path to save the results.
    - num_columns (int): Number of columns in the subplot grid.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from google.colab import files

    sorted_keys = sorted(grains.keys())
    num_grains = len(sorted_keys)
    num_rows = (num_grains + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, num_rows * 4))
    axes = axes.flatten()

    theta = np.linspace(0, 2 * np.pi, 100)

    for i, grain_id in enumerate(sorted_keys):
        ax = axes[i]
        grain_data = grains[grain_id]
        BW = grain_data['grain_bin_rs_PCD']

        ax.imshow(BW, cmap='gray')

        # Plot circles
        z = grain_data['z']
        r = grain_data['r']
        for center, radius in zip(z, r):
            ax.plot(center[0], center[1], 'o', markersize=12, markerfacecolor='none',
                    markeredgecolor='w', markeredgewidth=1)
            ax.plot(center[0], center[1], 'x', markersize=10, markerfacecolor='none',
                    markeredgecolor='w', markeredgewidth=1)
            ax.plot(np.cos(theta) * radius + center[0], np.sin(theta) * radius + center[1],
                    linewidth=1, color="lawngreen", linestyle="-")

        # Plot MCI
        R = grain_data['MCI'][2]
        cx = grain_data['MCI'][0]
        cy = grain_data['MCI'][1]
        ax.plot(np.cos(theta) * R + cx, np.sin(theta) * R + cy,
                linewidth=2, color="red", linestyle="-")

        # Plot convex points
        convex = grain_data['convex']
        ax.plot(convex[:, 0], convex[:, 1], 'o', markersize=4, markerfacecolor='none',
                markeredgecolor='r', markeredgewidth=1)

        # Add text annotations
        roundness = grain_data['Roundness']
        sphericity = grain_data['Roundness']  # Assuming sphericity is the same as Roundness
        d1 = grain_data['d1']
        d2 = grain_data['d2']
        ax.text(0.05, 0.95,
                f'Grain ID: {grain_id}\nRoundness, R: {roundness:.2f}\nSphericity: {sphericity:.2f}\nd1: {d1:.2f}\nd2: {d2:.2f}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'),
                fontdict={'style': 'italic'})

        ax.axis("off")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    # Save the figure
    parent_dir = os.path.basename(path)
    save_path = os.path.join(os.path.dirname(path), f"{parent_dir}_results.pdf")
    fig.savefig(save_path, dpi=300, format='pdf')
    print(f"Figure saved at {save_path}")
    plt.close(fig)

    # Download the PDF file in Colab
    files.download(save_path)

    # Save the data to an Excel file
    save_grains_to_excel(grains, 'grains_data.xlsx')

def save_grains_to_excel(grains, filename):
    """
    Saves grain metrics to an Excel file and downloads it in Colab.

    Parameters:
    - grains (dict): Dictionary containing all grains data.
    - filename (str): Name of the Excel file to save.
    """
    import pandas as pd
    from google.colab import files

    # Create a list to store data
    data = []

    # Iterate over each grain and extract specified columns
    for grain_id, grain_data in grains.items():
        row = {
            'Grain ID': grain_id,
            'Roundness': grain_data['Roundness'],
            'Sa': grain_data['Sa'],
            'Sc': grain_data['Sc'],
            'Sd': grain_data['Sd'],
            'Sp': grain_data['Sp'],
            'Swl': grain_data['Swl'],
            'AR': grain_data['AR'],
            'Cx': grain_data['Cx'],
            'd1': grain_data['d1'],
            'd2': grain_data['d2'],
            'de': grain_data['de'],
            'dins': grain_data['dins'],
            'dcir': grain_data['dcir'],
            'minf': grain_data['minf'],
            'maxf': grain_data['maxf'],
        }
        data.append(row)

    # Create a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    df.to_excel(filename, index=False)

    # Download the file in Colab
    files.download(filename)

def draw_circles_on_grain(grain_data):
    """
    Draws circles on the grain image and saves the result in 'grain_results' key.

    Parameters:
    - grain_data (dict): Dictionary containing grain data.

    Returns:
    - success (bool): True if successful, False otherwise.
    """
    import cv2
    import numpy as np
    from PIL import Image

    if 'grain_bin_rs_PCD' not in grain_data:
        print("Error: 'grain_bin_rs_PCD' key is missing in grain_data")
        return False

    # Load the grain image
    grain = grain_data['grain_bin_rs_PCD']

    # Check if grain is a PIL Image and convert it to a NumPy array
    if isinstance(grain, Image.Image):
        grain = np.array(grain)

    # Check if grain is a valid NumPy array
    if not isinstance(grain, np.ndarray):
        print("Error: 'grain_bin_rs_PCD' is not a valid NumPy array")
        return False

    # Convert the image to BGR format for OpenCV
    try:
        grain_bgr = cv2.cvtColor(grain, cv2.COLOR_GRAY2BGR)
    except cv2.error as e:
        print(f"Error converting image to BGR: {e}")
        return False

    # Extract necessary data from grain_data
    try:
        convex = grain_data['convex']
        Roundness = grain_data['Roundness']
        Sa = grain_data['Sa']
        Sd = grain_data['Sd']
        Sp = grain_data['Sp']
        Swl = grain_data['Swl']
        AR = grain_data['AR']
        Cx = grain_data['Cx']
        z = grain_data['z']
        r = grain_data['r']
        MCI = grain_data['MCI']
        MCC = grain_data['MCC']
    except KeyError as e:
        print(f"Error: Missing key {e} in grain_data")
        return False

    # Draw circles on the image
    for center, radius in zip(z, r):
        cv2.circle(grain_bgr, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)  # Green circles

    # Draw MCI circle
    cv2.circle(grain_bgr, (int(MCI[0]), int(MCI[1])), int(MCI[2]), (0, 0, 255), 2)  # Red MCI circle

    # Draw convex points
    for point in convex:
        cv2.circle(grain_bgr, (int(point[0]), int(point[1])), 0, (0, 0, 0), -1)  # Black convex points

    # Convert the result back to a PIL Image
    grain_bgr_pil = Image.fromarray(cv2.cvtColor(grain_bgr, cv2.COLOR_BGR2RGB))

    # Save the result in a new key
    grain_data['grain_results'] = grain_bgr_pil
    return True
