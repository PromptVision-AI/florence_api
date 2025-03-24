from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from PIL import Image
import io
from contextlib import asynccontextmanager

from app.core.model import load_model, cleanup_model, run_segmentation, run_object_detection, model
from app.utils.visualization import create_visualization, create_mask_image, create_detection_visualization
from app.services.cloudinary import configure_cloudinary, upload_image_to_cloudinary

# Configure Cloudinary
configure_cloudinary()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model and processor
    load_model()
    yield
    # Shutdown: Clean up resources
    cleanup_model()

app = FastAPI(title="Florence-2 Segmentation API", lifespan=lifespan)

@app.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
):
    """
    Segment objects in an image based on a text prompt.
    
    - **file**: The image file to process
    - **prompt**: Text description of the object to segment (e.g., "black dog")
    
    Returns:
    - Original image URL from Cloudinary
    - Mask image URL from Cloudinary
    - 2D numpy array representing the segmentation mask
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Upload original image to Cloudinary
        original_url = upload_image_to_cloudinary(contents)
        
        # Run segmentation
        segmentation_result = run_segmentation(image, prompt)
        
        # Create binary mask as 2D array
        mask_array = create_mask_image(image, segmentation_result)
        
        # Convert mask array to image for Cloudinary upload
        mask_image = Image.fromarray(mask_array.astype('uint8'))
        mask_buffer = io.BytesIO()
        mask_image.save(mask_buffer, format="PNG")
        mask_buffer.seek(0)
        
        # Upload mask to Cloudinary
        mask_url = upload_image_to_cloudinary(mask_buffer)
        
        # Prepare response
        response = {
            "success": True,
            "prompt": prompt,
            "original_image_url": original_url,
            "mask_url": mask_url
        }
        
        return response
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/segment/visualize")
async def segment_image_visualize(
    file: UploadFile = File(...),
    prompt: str = Form(...),
):
    """
    Segment objects in an image and return a visualization.
    
    - **file**: The image file to process
    - **prompt**: Text description of the object to segment (e.g., "black dog")
    
    Returns a PNG image with the visualization of the segmentation.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run segmentation
        segmentation_result = run_segmentation(image, prompt)
        
        # Create visualization
        vis_image = create_visualization(image, segmentation_result)
        
        # Convert visualization to bytes
        img_byte_arr = io.BytesIO()
        vis_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    prompt: str = Form(...),
):
    """
    Detect objects in an image based on a text prompt.
    
    - **file**: The image file to process
    - **prompt**: Text description of the object to detect (e.g., "black dog")
    
    Returns:
    - Original image URL from Cloudinary
    - Annotated image URL from Cloudinary (with bounding boxes and centroids)
    - List of bounding boxes [x1, y1, x2, y2]
    - List of centroids [cx, cy] for each box
    - List of labels for each detected object
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Upload original image to Cloudinary
        original_url = upload_image_to_cloudinary(contents)
        
        # Run object detection
        detection_result = run_object_detection(image, prompt)
        
        # Create visualization with bounding boxes and centroids
        vis_image = create_detection_visualization(image, detection_result)
        
        # Convert visualization to bytes for Cloudinary upload
        vis_buffer = io.BytesIO()
        vis_image.save(vis_buffer, format="PNG")
        vis_buffer.seek(0)
        
        # Upload visualization to Cloudinary
        vis_url = upload_image_to_cloudinary(vis_buffer)
        
        # Prepare response
        response = {
            "success": True,
            "prompt": prompt,
            "original_image_url": original_url,
            "annotated_image_url": vis_url,
            "bounding_boxes": detection_result['bboxes'],
            "centroids": detection_result['centroids'],
            "labels": detection_result['labels']
        }
        
        return response
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None} 