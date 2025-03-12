from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import io

def create_visualization(image, segmentation_result, save_path=None):
    """Create a visualization of the segmentation result."""
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Create and plot mask
    mask = Image.new('L', (image.width, image.height), 0)
    draw = Image.new('RGBA', (image.width, image.height), (0, 0, 0, 0))
    draw_ctx = ImageDraw.Draw(draw)
    
    # Draw each polygon with a different color
    colors = [(255,0,0,128), (0,255,0,128), (0,0,255,128), (255,255,0,128), (255,0,255,128)]
    for i, (polygons, label) in enumerate(zip(segmentation_result['polygons'], segmentation_result['labels'])):
        color = colors[i % len(colors)]
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                continue
            _polygon = _polygon.reshape(-1).tolist()
            draw_ctx.polygon(_polygon, fill=color)
            # Add label text
            centroid = np.mean(_polygon[::2]), np.mean(_polygon[1::2])
            draw_ctx.text((centroid[0], centroid[1]), label, fill=(255,255,255,255))
    
    # Plot segmentation mask
    ax2.imshow(draw)
    ax2.set_title("Segmentation Mask")
    ax2.axis('off')
    
    # Plot overlay
    overlay = Image.alpha_composite(image.convert('RGBA'), draw)
    ax3.imshow(overlay)
    ax3.set_title("Overlay")
    ax3.axis('off')
    
    plt.tight_layout()
    
    # Save or return the visualization
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return save_path
    else:
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return Image.open(buf)

def create_mask_image(image, segmentation_result):
    """Create a binary mask as a 2D numpy array."""
    # Create a binary mask (black background)
    mask = Image.new('L', (image.width, image.height), 0)
    draw_ctx = ImageDraw.Draw(mask)
    
    # Draw each polygon as white
    for polygons, _ in zip(segmentation_result['polygons'], segmentation_result['labels']):
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                continue
            
            # Convert polygon to list format for PIL
            _polygon = _polygon.reshape(-1).tolist()
            
            # Draw the polygon filled with white (255)
            draw_ctx.polygon(_polygon, fill=255)  # Pure white
    
    # Convert to numpy array
    mask_array = np.array(mask)
    return mask_array 