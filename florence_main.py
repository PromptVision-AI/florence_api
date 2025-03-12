from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import matplotlib.patches as patches 
import matplotlib.pyplot as plt
import numpy as np

model_id = '/home/diego/Documents/master/S4/Industry_P/codes/florence2/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id,trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def run_example(task_prompt,imageOriginal,text_input=None):
    if  text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt,images=imageOriginal,return_tensors="pt").to('cuda',torch.float16)

    generated_ids = model.generate(
        input_ids=inputs['input_ids'].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=5,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size = (imageOriginal.width, imageOriginal.height)
    )
    return parsed_answer

def plot_bbox(image, data):
   # Create a figure and axes  
    fig, ax = plt.subplots()  
      
    # Display the image  
    ax.imshow(image)  
      
    # Plot each bounding box  
    for bbox, label in zip(data['bboxes'], data['labels']):  
        # Unpack the bounding box coordinates  
        x1, y1, x2, y2 = bbox  
        # Create a Rectangle patch  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
        # Add the rectangle to the Axes  
        ax.add_patch(rect)  
        # Annotate the label  
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
      
    # Remove the axis ticks and labels  
    ax.axis('off')  
      
    # Show the plot  
    plt.show()  



from PIL import Image, ImageDraw, ImageFont 
import random
import numpy as np
colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']
def draw_polygons(image, prediction, fill_mask=False):  
    # Load the image  
   
    draw = ImageDraw.Draw(image)  
      
   
    # Set up scale factor if needed (use 1 if not scaling)  
    scale = 1  
      
    # Iterate over polygons and labels  
    for polygons, label in zip(prediction['polygons'], prediction['labels']):  
        color = random.choice(colormap)  
        fill_color = random.choice(colormap) if fill_mask else None  
          
        for _polygon in polygons:  
            _polygon = np.array(_polygon).reshape(-1, 2)  
            if len(_polygon) < 3:  
                print('Invalid polygon:', _polygon)  
                continue  
              
            _polygon = (_polygon * scale).reshape(-1).tolist()  
              
            # Draw the polygon  
            if fill_mask:  
                draw.polygon(_polygon, outline=color, fill=fill_color)  
            else:  
                draw.polygon(_polygon, outline=color)  
              
            # Draw the label text  
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)  
  
    # Save or display the image  
    image.show()  # Display the image  
    

def remove_object_bbox(image, prediction):
    """Removes the detected objects by setting their bounding box area to black."""
    draw = ImageDraw.Draw(image)

    for polygons in prediction['polygons']:
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)  # Convert to NumPy array
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue
            
            # Get bounding box (min x, min y, max x, max y)
            x_min, y_min = np.min(_polygon, axis=0)
            x_max, y_max = np.max(_polygon, axis=0)

            # Draw the bounding box as a black rectangle
            draw.rectangle([x_min, y_min, x_max, y_max], fill=(0, 0, 0))
    
    return image  # Return the modified image


loadedImage = Image.open('exa.png')

imgPlot = np.asarray(loadedImage)
plt.imshow(imgPlot)
plt.show()

# taskPromptInput1 = "<OD>"
# results = run_example(taskPromptInput1,loadedImage)
# plot_bbox(imgPlot, results['<OD>'])

# print('********************************')
# print('********************************')

text_input = "black dog"
taskPromptInput2 = '<CAPTION_TO_PHRASE_GROUNDING>'
results = run_example(taskPromptInput2,loadedImage, text_input=text_input)
plot_bbox(imgPlot, results['<CAPTION_TO_PHRASE_GROUNDING>'])

# print('********************************')
# print('********************************')

taskPromptInput3 = '<REFERRING_EXPRESSION_SEGMENTATION>'

for i in range(len(results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'])):
    results = run_example(taskPromptInput3,loadedImage, text_input=text_input)
    draw_polygons(loadedImage, results['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask=True)  
    loadedImage = remove_object_bbox(loadedImage, results['<REFERRING_EXPRESSION_SEGMENTATION>'])
    
 