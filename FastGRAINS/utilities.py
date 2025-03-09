
from PIL import Image
import cv2
import numpy as np

def resize_image(image, scale_factor, method):
    """
    Resizes an image by a scale factor using the specified resampling method.

    Parameters:
    - image (PIL.Image.Image): Input image.
    - scale_factor (float): Scaling factor.
    - method (int): Resampling method (e.g., Image.LANCZOS).

    Returns:
    - resized_image (PIL.Image.Image): Resized image.
    """
    new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
    resized_image = image.resize(new_size, method)
    return resized_image

def process_binary_image(image, reduction_factor):
    """
    Processes a binary image by applying blurring and thresholding.

    Parameters:
    - image (PIL.Image.Image): Input binary image.
    - reduction_factor (float): Reduction factor.

    Returns:
    - processed_image (PIL.Image.Image): Processed binary image.
    """
    image = np.array(image).astype(np.uint8)  # Ensure image is uint8
    image = cv2.GaussianBlur(image, (3, 3), 0)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
import os
import numpy as np
import io
import matplotlib.pyplot as plt
from functools import partial
import gc
from constants import IMAGES_DIR, OUTPUT_DIR

# Global variables to store grain data for processing
filtered_grains = {}
# OUTPUT_DIR = os.path.join('/content', 'output_images')
def create_rgb_and_binary_grids(grains, output_rgb_path, output_binary_path, background_color='black'):
    """
    Creates an interactive UI for selecting grains and generating image grids.
    
    Args:
        grains: Dictionary of grain data
        output_rgb_path: Path to save RGB grid image
        output_binary_path: Path to save binary grid image
        background_color: Color for RGB grid background ('black' or 'white')
    
    Returns:
        None (displays interactive widgets and saves images when selection is complete)
    """
    global filtered_grains  # Access the global variable
    if not grains:
        print("❌ No grains available for processing! Please run the segmentation first.")
        return
    filtered_grains = grains.copy()  # Initialize with a copy of the input grains
    
    # Build list of images with IDs and sizes
    image_list = [
        {
            'id': grain_id,
            'img_rgb': grain_data.get('grain_rs', None),
            'img_bin': grain_data.get('grain_bin_rs', None),
            'size': (grain_data.get('grain_rs').width * grain_data.get('grain_rs').height 
                    if grain_data.get('grain_rs') else 0),
            'metrics': grain_data.get('metrics', {})
        }
        for grain_id, grain_data in grains.items() 
        if grain_data.get('grain_rs')
    ]
    
    # Sort images by size (largest first)
    image_list.sort(key=lambda x: x['size'], reverse=True)
    
    if not image_list:
        print("❌ No valid grain images found!")
        return
    
    # Progress indicator
    progress = widgets.IntProgress(
        value=0,
        min=0,
        max=len(image_list),
        description='Loading:',
        bar_style='info',
        orientation='horizontal'
    )
    display(progress)
    
    # Create widgets for filter controls
    min_size_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=100,
        step=5,
        description='Min Size %:',
        layout=widgets.Layout(width='300px')
    )
    
    max_size_slider = widgets.IntSlider(
        value=100,
        min=0,
        max=100,
        step=5,
        description='Max Size %:',
        layout=widgets.Layout(width='300px')
    )
    
    select_all_btn = widgets.Button(
        description='Select All',
        button_style='info',
        layout=widgets.Layout(width='120px')
    )
    
    deselect_all_btn = widgets.Button(
        description='Deselect All',
        button_style='warning',
        layout=widgets.Layout(width='120px')
    )
    
    filter_output = widgets.Output()
    display(widgets.HBox([min_size_slider, max_size_slider, select_all_btn, deselect_all_btn]))
    display(filter_output)
    
    # Create containers for image widgets
    image_container = widgets.Output()
    display(image_container)
    
    # Container for all checkboxes and their data
    checkboxes = []
    
    # Function to update the image display based on filters
    def update_image_display():
        if not checkboxes:
            return
        
        with image_container:
            clear_output(wait=True)
            
            # Get size values for filtering
            min_size_pct = min_size_slider.value / 100
            max_size_pct = max_size_slider.value / 100
            
            # Find the maximum size for percentage calculation
            max_size = max(item['data']['size'] for item in checkboxes)
            
            # Filter items based on size
            visible_items = []
            for item in checkboxes:
                size_pct = item['data']['size'] / max_size
                if min_size_pct <= size_pct <= max_size_pct:
                    item['widget'].layout.display = 'block'
                    visible_items.append(item['widget'])
                else:
                    item['widget'].layout.display = 'none'
            
            # Create a grid layout for the visible items
            rows = []
            current_row = []
            for i, item in enumerate(visible_items):
                current_row.append(item)
                if len(current_row) == 4 or i == len(visible_items) - 1:
                    rows.append(widgets.HBox(current_row))
                    current_row = []
            
            grid = widgets.VBox(rows)
            display(grid)
            
            # Show count of visible items
            print(f"Showing {len(visible_items)} of {len(checkboxes)} grains")
    
    # Function to handle select/deselect all
    def select_all(b):
        for item in checkboxes:
            item['checkbox'].value = True
    
    def deselect_all(b):
        for item in checkboxes:
            item['checkbox'].value = False
    
    select_all_btn.on_click(select_all)
    deselect_all_btn.on_click(deselect_all)
    
    # Connect filter controls to update function
    min_size_slider.observe(lambda x: update_image_display(), names='value')
    max_size_slider.observe(lambda x: update_image_display(), names='value')
    
    # Process each image and create widgets
    all_widgets = []
    for i, item in enumerate(image_list):
        img_rgb = item['img_rgb']
        img_bin = item['img_bin']
        
        # Create thumbnails
        img_rgb_thumbnail = img_rgb.copy()
        img_rgb_thumbnail.thumbnail((180, 180))
        
        # Handle binary image (create white image if None)
        if img_bin:
            img_bin_thumbnail = img_bin.copy()
        else:
            img_bin_thumbnail = Image.new('RGB', img_rgb.size, 'white')
        img_bin_thumbnail.thumbnail((180, 180))
        
        # Combine thumbnails side by side
        combined = Image.new(
            'RGB',
            (img_rgb_thumbnail.width + img_bin_thumbnail.width + 10,  # Add some padding
             max(img_rgb_thumbnail.height, img_bin_thumbnail.height))
        )
        combined.paste(img_rgb_thumbnail, (0, 0))
        combined.paste(img_bin_thumbnail, (img_rgb_thumbnail.width + 10, 0))
        
        # Convert to bytes for widget
        buffer = io.BytesIO()
        combined.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Create a checkbox and an image widget
        checkbox = widgets.Checkbox(
            value=True,
            description=f"Grain {item['id']}",
            indent=False
        )
        
        # Extract some metrics if available
        metrics_text = ""
        if item['metrics']:
            try:
                metrics_text = f"AR: {item['metrics'].get('aspect_ratio', 'N/A'):.2f} | "
                metrics_text += f"R: {item['metrics'].get('roundness', 'N/A'):.2f} | "
                metrics_text += f"S: {item['metrics'].get('sphericity', 'N/A'):.2f}"
            except (TypeError, AttributeError, ValueError):
                metrics_text = "Metrics N/A"
        
        # Create a label with metrics
        metrics_label = widgets.HTML(
            value=f"<small>{metrics_text}</small>"
        )
        
        img_widget = widgets.Image(
            value=buffer.getvalue(),
            format='png',
            width=360,
            height=180
        )
        
        # Combine checkbox, metrics and image in a box
        grain_box = widgets.VBox([
            widgets.HBox([checkbox, metrics_label]),
            img_widget
        ], layout=widgets.Layout(
            margin='5px',
            border='1px solid #ddd',
            padding='5px'
        ))
        
        # Add to tracking containers
        checkboxes.append({
            'checkbox': checkbox,
            'widget': grain_box,
            'data': {
                'id': item['id'],
                'img_rgb': img_rgb,
                'img_bin': img_bin,
                'size': item['size']
            }
        })
        
        # Update progress
        progress.value = i + 1
    
    # Initial display update
    update_image_display()
    
    # Create and display the Generate Grids button
    generate_button = widgets.Button(
        description='✅ GENERATE GRIDS',
        button_style='success',
        layout=widgets.Layout(width='200px', height='60px', font_weight='bold'),
        style={'font_size': '16px'}
    )
    display(generate_button)
    
    # Output area for generation results
    generation_output = widgets.Output()
    display(generation_output)
    
    def on_generate_click(b):
        global filtered_grains  # Access the global variable
        
        with generation_output:
            clear_output(wait=True)
            print("🔄 Processing selected grains...")
            
            # Gather selected items from the checkboxes
            selected = [item for item in checkboxes if item['checkbox'].value]
            
            if not selected:
                print("⚠️ No grains selected! Please select at least one grain.")
                return
            
            print(f"✓ Found {len(selected)} selected grains")
            
            # Sort by size (largest first)
            selected.sort(key=lambda x: x['data']['size'], reverse=True)
            
            # Get the selected IDs
            selected_ids = [item['data']['id'] for item in selected]
            
            # Update the global filtered_grains with only the selected keys
            filtered_grains = {k: grains[k] for k in selected_ids if k in grains}
            print(f"✓ Filtered {len(filtered_grains)} grains for processing")
            
            # Build the grid images
            try:
                # Calculate dimensions
                num_cols = min(len(selected), 6)  # Max 6 columns
                cell_size = max(
                    max(item['data']['img_rgb'].width, item['data']['img_rgb'].height)
                    for item in selected
                )
                
                # Ensure minimum cell size for visibility
                cell_size = max(cell_size, 200)
                
                # Calculate rows needed
                num_rows = (len(selected) + num_cols - 1) // num_cols
                
                # Set background colors
                bg_color = (0, 0, 0) if background_color == 'black' else (255, 255, 255)
                
                # Create new images for grids
                grid_img = Image.new('RGB', (cell_size * num_cols, cell_size * num_rows), bg_color)
                grid_binary_img = Image.new('RGB', (cell_size * num_cols, cell_size * num_rows), (255, 255, 255))
                
                # Progress for grid generation
                grid_progress = widgets.IntProgress(
                    value=0,
                    min=0,
                    max=len(selected),
                    description='Building:',
                    bar_style='info',
                    orientation='horizontal'
                )
                display(grid_progress)
                
                # Paste each image into the grid
                for idx, item in enumerate(selected):
                    row, col = divmod(idx, num_cols)
                    x, y = col * cell_size, row * cell_size
                    
                    # Get image data
                    rgb_img = item['data']['img_rgb']
                    bin_img = item['data']['img_bin']
                    
                    # Paste RGB image with proper centering
                    rgb_bg = Image.new('RGB', (cell_size, cell_size), bg_color)
                    rgb_bg.paste(
                        rgb_img, 
                        ((cell_size - rgb_img.width) // 2,
                         (cell_size - rgb_img.height) // 2)
                    )
                    grid_img.paste(rgb_bg, (x, y))
                    
                    # Paste binary image with proper centering
                    bin_bg = Image.new('RGB', (cell_size, cell_size), (255, 255, 255))
                    if bin_img:
                        bin_bg.paste(
                            bin_img, 
                            ((cell_size - bin_img.width) // 2,
                             (cell_size - bin_img.height) // 2)
                        )
                    grid_binary_img.paste(bin_bg, (x, y))
                    
                    # Update progress
                    grid_progress.value = idx + 1
                
                # Save the grid images
                os.makedirs(os.path.dirname(output_rgb_path), exist_ok=True)
                grid_img.save(output_rgb_path)
                grid_binary_img.save(output_binary_path)
                
                # Display preview of the grids
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(np.array(grid_img))
                axes[0].set_title("RGB Grid")
                axes[0].axis('off')
                
                axes[1].imshow(np.array(grid_binary_img))
                axes[1].set_title("Binary Grid")
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.show()
                
                print(f"💾 Grids successfully saved to:")
                print(f"📁 RGB Grid: {output_rgb_path}")
                print(f"📁 Binary Grid: {output_binary_path}")
                print(f"🎉 Selected {len(selected_ids)} grains have been processed and saved!")
                
            except Exception as e:
                print(f"❌ Error generating grids: {e}")
                import traceback
                traceback.print_exc()
                
            # Clean up to free memory
            gc.collect()
    
    generate_button.on_click(on_generate_click)

# Example usage (assuming grains dictionary is populated)
def run_grain_selection(grains_dict=None):
    """Run the grain selection workflow with the provided grains dictionary"""
    if grains_dict is None:
        print("⚠️ No grains available. Please run segmentation first.")
        return
    
    # Define output paths
    rgb_path = os.path.join(OUTPUT_DIR, 'grid_rgb.png')
    bin_path = os.path.join(OUTPUT_DIR, 'grid_bin.png')
    
    # Run the selection UI
    create_rgb_and_binary_grids(grains_dict, rgb_path, bin_path, background_color='black')
