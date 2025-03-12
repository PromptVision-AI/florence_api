import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from app.core.config import MODEL_PATH
import numpy as np

# Global variables for model and processor
model = None
processor = None

def load_model():
    """Load the Florence-2 model and processor."""
    global model, processor
    print("Loading Florence-2 model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True, 
        torch_dtype='auto'
    ).eval().cuda()
    
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True
    )
    print("Model loaded successfully!")

def cleanup_model():
    """Clean up model resources."""
    global model, processor
    if model is not None:
        model.cpu()
        del model
    if processor is not None:
        del processor
    torch.cuda.empty_cache()

def run_segmentation(image, prompt):
    """Run the Florence-2 model for segmentation."""
    # Prepare task prompt for referring expression segmentation
    task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
    
    # Process the input
    inputs = processor(
        text=task_prompt + prompt,
        images=image,
        return_tensors="pt"
    ).to('cuda', torch.float16)
    
    # Generate output
    generated_ids = model.generate(
        input_ids=inputs['input_ids'].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=5,
    )
    
    # Decode and post-process
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    return parsed_answer['<REFERRING_EXPRESSION_SEGMENTATION>']

def run_object_detection(image, prompt):
    """Run the Florence-2 model for object detection.
    
    Args:
        image: PIL Image to process
        prompt: Text description of the object to detect
        
    Returns:
        dict: Dictionary containing:
            - bboxes: List of bounding boxes [x1, y1, x2, y2]
            - labels: List of labels for each box
            - centroids: List of centroids [cx, cy] for each box
    """
    # Prepare task prompt for object detection
    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    
    # Process the input
    inputs = processor(
        text=task_prompt + prompt,
        images=image,
        return_tensors="pt"
    ).to('cuda', torch.float16)
    
    # Generate output
    generated_ids = model.generate(
        input_ids=inputs['input_ids'].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=5,
    )
    
    # Decode and post-process
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    result = parsed_answer['<CAPTION_TO_PHRASE_GROUNDING>']
    
    # Calculate centroids for each bounding box
    centroids = []
    for bbox in result['bboxes']:
        x1, y1, x2, y2 = bbox
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        centroids.append([centroid_x, centroid_y])
    
    # Add centroids to the result
    result['centroids'] = centroids
    
    return result 